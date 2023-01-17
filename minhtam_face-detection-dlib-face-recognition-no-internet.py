!pip install '/kaggle/input/dlibpkg/dlib-19.19.0'

!pip install '/kaggle/input/face-recognition/face_recognition_models-0.3.0/face_recognition_models-0.3.0'

!pip install '/kaggle/input/face-recognition/face_recognition-0.1.5-py2.py3-none-any.whl'



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import face_recognition

import cv2

import matplotlib.pyplot as plt

v_cap = cv2.VideoCapture('/kaggle/input/deepfake-detection-challenge/test_videos/adohdulfwb.mp4')

for j in range(1):

    success, vframe = v_cap.read()
vframe = cv2.cvtColor(vframe,cv2.COLOR_BGR2RGB)

plt.imshow(vframe)
 

face_positions = face_recognition.face_locations(vframe)
for face_position in face_positions:

    y0,x1,y1,x0 = face_position

    img = cv2.rectangle(vframe,(x0,y0),(x1,y1),(255,0,0),5)

plt.imshow(img)
