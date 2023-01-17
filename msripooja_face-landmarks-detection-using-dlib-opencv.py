!pip install dlib
!pip install imutils
import os

import cv2

import dlib

import imutils

from imutils import face_utils

import matplotlib.pyplot as plt



image_path = "../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/Chip_Knight/Chip_Knight_0001.jpg"

filter_path = "../input/dlib-landmarks-predictor/shape_predictor_68_face_landmarks.dat"



print(dlib.__version__)

print(imutils.__version__)
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(filter_path)
image = cv2.imread(image_path)

image = imutils.resize(image, width=500)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

 

# detect faces in the image

rects = detector(image, 1)
for (i, rect) in enumerate(rects):

    shape = predictor(image, rect)

    shape = face_utils.shape_to_np(shape)

    

    (x, y, w, h) = face_utils.rect_to_bb(rect)

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)



    for (x, y) in shape:

        cv2.circle(image, (x, y), 1, (0, 255, 0), 3)



plt.figure(figsize=(10,10))

plt.imshow(image)

plt.xticks([])

plt.yticks([])

plt.show()