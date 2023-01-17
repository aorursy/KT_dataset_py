!pip install '/kaggle/input/dlibpkg/dlib-19.19.0'
import numpy as np

import cv2

from matplotlib import pyplot as plt

import dlib
%%time

sample = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/aagfhgtpmv.mp4'



reader = cv2.VideoCapture(sample)

_, image = reader.read()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



face_detector = dlib.get_frontal_face_detector()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_detector(gray, 1)

if len(faces) > 0:

    face = faces[0]

    

face_image = image[face.top():face.bottom(), face.left():face.right()]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

ax1.imshow(image)

ax1.xaxis.set_visible(False)

ax1.yaxis.set_visible(False)



ax2.imshow(face_image)

ax2.xaxis.set_visible(False)

ax2.yaxis.set_visible(False)



plt.grid(False)

plt.tight_layout()