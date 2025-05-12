"""
Driver Drowsiness Detection System
-----------------------------------

1. Project Title
• Driver Drowsiness Detection System

2. Problem Statement / Objective
• Detect if a driver is drowsy (eyes closed for a prolonged period) using a live webcam feed.
• Alert the driver via audio if signs of drowsiness are detected.

3. Input Data Source
• Live video stream (captured using a webcam)

4. Preprocessing Steps
• Flip the frame horizontally (mirror effect)
• Convert BGR to RGB for MediaPipe processing
• Apply Gaussian Blur to reduce noise

5. Feature Extraction / Detection
• Face and eye landmarks detection using MediaPipe Face Mesh
• Eye Aspect Ratio (EAR) calculation to detect closed eyes

6. Computer Vision Technique
• Recognition: Eye closure detection based on facial landmarks
• Optional tracking via consistent face landmark indexing

7. Output / Result Display
• Annotated video frames showing:
    - Eye rectangles
    - Face bounding box
    - Alerts (text: "WARNING" / "ALERT: WAKE UP!")
• Save snapshot images when drowsiness is detected
• Save sleep log in CSV
• Generate sleep duration analysis plot at session end

8. User Interaction
• Real-time feedback on screen with alerts
• Sound alerts via pygame
• Press "ESC" to end the session
"""

import cv2
import mediapipe as mp
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import pygame
from threading import Thread

# ===== Initialization =====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

pygame.mixer.init()
warning_sound = pygame.mixer.Sound('mixkit-confirmation-tone-2867.wav')
alert_sound = pygame.mixer.Sound('Ya marary .mp3')


dark_pixel_threshold = 30     
dark_pixel_count_limit = 2000  
dark_frame_moments = 0




# ===== Eye Aspect Ratio Calculation =====
def eye_aspect_ratio(eye_landmarks):
    import math
    def distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    vertical1 = distance(eye_landmarks[1], eye_landmarks[5])
    vertical2 = distance(eye_landmarks[2], eye_landmarks[4])
    horizontal = distance(eye_landmarks[0], eye_landmarks[3])
    
    return (vertical1 + vertical2) / (2.0 * horizontal)

# ===== Video Capture =====
cap = cv2.VideoCapture(0)

eye_closed = False
start_time = 0
sleep_started_at = None
warning_played = False
alert_played = False

total_sleep_time = 0 
sleep_periods = []

# ===== CSV Logging =====
csv_file = open('sleep_log.csv', mode='a', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Sleep Start Time", "Sleep End Time", "Sleep Duration (seconds)"])

# ===== Thresholds =====
EYE_CLOSED_THRESHOLD = 0.25
WARNING_TIME = 3    
ALERT_TIME = 5

# ===== Interactive UI Setup =====
# def draw_text(frame, text, position, color=(255, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, size=1, thickness=2):
#     cv2.putText(frame, text, position, font, size, color, thickness)

# def draw_button(frame, text, position, width=200, height=50, color=(0, 255, 0)):
#     cv2.rectangle(frame, position, (position[0] + width, position[1] + height), color, -1)
#     draw_text(frame, text, (position[0] + 50, position[1] + 30), color=(0, 0, 0))


total_dark_pixels = 0
frame_count = 0

# ===== Main Loop =====
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Cannot capture frame.")
        break

    # ========== Preprocessing ==========
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray < dark_pixel_threshold)

    if dark_pixels > dark_pixel_count_limit:
        dark_frame_moments += 1

    total_dark_pixels += dark_pixels
    frame_count += 1

    if dark_pixels > dark_pixel_count_limit:
            dark_frame_moments += 1

            # Show visual warning on frame
            cv2.putText(frame, "DARK AREA DETECTED!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # # Play a sound alert if not already playing
            # if not pygame.mixer.Channel(1).get_busy():
            #     pygame.mixer.Channel(1).play(pygame.mixer.Sound('dark_alert.wav'))  # Make sure this sound exists

    else:
        dark_frame_moments = 0  # Reset if lighting improves

    

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to YUV and apply CLAHE on luminance
    yuv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    rgb_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    rgb_frame = cv2.GaussianBlur(rgb_frame, (5, 5), 0)

    results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # ========== Eye Landmarks ==========
        left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

        # ========== Eye Closure Detection ==========
        left_ratio = eye_aspect_ratio(left_eye)
        right_ratio = eye_aspect_ratio(right_eye)
        avg_ratio = (left_ratio + right_ratio) / 2.0

        if avg_ratio < EYE_CLOSED_THRESHOLD:
            if not eye_closed:
                start_time = time.time()
                sleep_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
                eye_closed = True
                if warning_played:
                    warning_sound.stop()
                    warning_played = False
                if alert_played:
                    alert_sound.stop()
                    alert_played = False

            else:
                elapsed = time.time() - start_time

                if elapsed > ALERT_TIME and not alert_played:
                    if warning_played:
                        warning_sound.stop()
                        warning_played = False
                    alert_sound.play(-1)
                    alert_played = True

                    # ========== Saving Image with Description ==========
                    all_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                    x_coords = [pt[0] for pt in all_points]
                    y_coords = [pt[1] for pt in all_points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    cropped_face = frame[y_min:y_max, x_min:x_max]
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                    sleep_duration = round(elapsed, 2)
                    image_name = f"drowsy_face_{timestamp}.jpg"

                    padded_image = cv2.copyMakeBorder(
                        cropped_face, top=50, bottom=50, left=60, right=60,
                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )

                    cv2.putText(
                        padded_image,
                        f"ALERT: Slept for {sleep_duration}s at {timestamp}",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2
                    )
                    cv2.imwrite(image_name, padded_image)
                    print(f"[INFO] Drowsy face saved as {image_name}")

                elif elapsed > WARNING_TIME and not warning_played and not alert_played:
                    warning_sound.play(-1)
                    warning_played = True

                if elapsed > WARNING_TIME:
                    cv2.putText(frame, "WARNING: Eyes closed!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                if elapsed > ALERT_TIME:
                    cv2.putText(frame, "ALERT: WAKE UP!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        else:
            if eye_closed:
                sleep_end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                sleep_duration = time.time() - start_time
                if sleep_duration > WARNING_TIME:
                    csv_writer.writerow([sleep_started_at, sleep_end_time, round(sleep_duration, 2)])
                    total_sleep_time += sleep_duration
                    sleep_periods.append((start_time, time.time()))
                    csv_file.flush()
                eye_closed = False

        # ========== Visualization ==========
        for eye in [left_eye, right_eye]:
            x_coords = [pt[0] for pt in eye]
            y_coords = [pt[1] for pt in eye]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)



        # Face rectangle
        all_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        x_coords = [pt[0] for pt in all_points]
        y_coords = [pt[1] for pt in all_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Background blur
        mask = np.zeros_like(frame)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = np.where(mask == np.array([255, 255, 255]), frame, blurred)

        cv2.putText(frame, "Driver Detected", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # ========== Display ==========
    screen_width = 1000
    screen_height = 750

    # frame = cv2.resize(frame, (screen_width, screen_height))

    cv2.namedWindow('Driver Monitoring', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Driver Monitoring', screen_width, screen_height)
    cv2.imshow('Driver Monitoring', frame)


    hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
    

    for x, y in enumerate(hist):
        cv2.line(hist_img, (x, 300), (x, 300 - int(y)), (255, 255, 255))

    cv2.imshow("Histogram", hist_img)


    if cv2.waitKey(5) & 0xFF == 27:
        break

# ===== Session End Summary =====
cap.release()
csv_file.close()
print(f"Total Sleep Time During Session: {round(total_sleep_time, 2)} seconds.")
print(f"\nTotal very dark moments detected: {dark_frame_moments}")

if frame_count > 0:
    avg_dark_pixels = total_dark_pixels / frame_count
    print(f"Average number of dark pixels per frame: {avg_dark_pixels:.2f}")


# ===== Plotting Sleep Analysis =====
if sleep_periods:
    start_times = [start for start, end in sleep_periods]
    durations = [end - start for start, end in sleep_periods]

    plt.figure(figsize=(12, 6))
    for start, duration in zip(start_times, durations):
        color = 'orange' if duration < 10 else 'red'
        plt.bar(start, duration, width=2, color=color)

    plt.xlabel('Time (seconds since start)')
    plt.ylabel('Sleep Duration (seconds)')
    plt.title('Driver Sleep Periods During Session')
    plt.grid(True)
    plt.legend(['Orange: Warning (<10s)', 'Red: Danger (>=10s)'])
    plt.savefig('detailed_analysis.png')
    print("Plot saved as 'detailed_analysis.png'.")
else:
    print("No sleep periods recorded.")

cv2.destroyAllWindows()
