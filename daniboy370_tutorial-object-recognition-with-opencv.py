# Install required libraries

!pip3 install --upgrade pip

!pip install moviepy

!pip install pytube3



# Import libraries

import os

import cv2

import moviepy

import numpy as np 

from pytube import YouTube

from datetime import timedelta

from IPython.display import Video

from IPython.display import YouTubeVideo

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip



URL_YouTube = 'https://www.youtube.com/watch?v=d-souLImH-I'



def YouTube_time_window(url, **kwargs):

    id_ = url.split("=")[-1]

    return YouTubeVideo(id_, **kwargs)



# enter time window [hr;mn;sec]

time_start = int(timedelta(hours=0, minutes=0, seconds=57).total_seconds())



# enter Duration [sec]

time_duration = 25



# Launch

time_end = time_start + time_duration

width, height = 550, 350

YouTube_time_window(URL_YouTube, start=time_start, width=width, height=height)

# Download and rename files 

Video_file_raw = YouTube(URL_YouTube).streams.first().download()

DIR_vid = '/kaggle/working/'

Video_file = DIR_vid + 'Vid_in.mp4'

os.rename(Video_file_raw, Video_file)



# Extract desired segment 

Video_out_name = 'Vid_out.mp4'

ffmpeg_extract_subclip(Video_file, time_start, time_end, targetname=Video_out_name)

Video(Video_out_name, width=width, height=height)
def region_of_interest(img, vertices):

    mask = np.zeros_like(img)

    match_mask_color = 255

    cv2.fillPoly(mask, vertices, match_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image



def draw_the_lines(img, lines):

    img = np.copy(img)

    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)



    for line in lines:

        for x1, y1, x2, y2 in line:

            cv2.line(blank_image, (x1,y1), (x2,y2), (255, 0, 0), thickness=5)



    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)

    return img



def process(image):

    height = image.shape[0]

    width = image.shape[1]

    region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    canny_image = cv2.Canny(gray_image, 100, 120)

    cropped_image = region_of_interest(canny_image,

                                       np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=50,

                           lines=np.array([]), minLineLength=40, maxLineGap=100)

    image_with_lines = draw_the_lines(image, lines)

    return image_with_lines
cap = cv2.VideoCapture(DIR_vid + 'Vid_out.mp4')

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

vid_out = cv2.VideoWriter(DIR_vid + 'Vid_Mask.mp4', fourcc, 20, (int(cap.get(3)), int(cap.get(4))))



# ----------------- Initialization ----------------- #

_, img_org = cap.read()

frame1 = img_org; frame2 = frame1

frame_i, frame_tot = 0, 480



while cap.isOpened() & (frame_i < frame_tot):

    if (frame_i%25 == 0):

        print('frame : %d / %d'% (frame_i, frame_tot) )

    frame_i = frame_i + 1 # frame_idx

    # ------- Hough Transform for alignment -------- #

    frame2 = process(frame2)

    # cv2_imshow(frame2)



    # -------------- Object Detection -------------- #

    diff = cv2.absdiff(frame1, frame2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (29, 29), 0)

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    for contour in contours:

        (x, y, w, h) = cv2.boundingRect(contour)



        if cv2.contourArea(contour) < 500:

            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)



    cv2.drawContours(frame1, contours, -1, (0, 0, 255), 1)

    vid_out.write(frame1)

    # cv2_imshow(frame1)



    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



    # ------------- Towards next step ------------- #

    frame1 = frame2

    _, frame2 = cap.read()

    # frame2 = cv.resize(frame2, low_res) # uncomment to resize resolution



cap.release()

vid_out.release()

cv2.destroyAllWindows()