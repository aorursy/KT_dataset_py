import numpy as np

import pandas as pd

import tensorflow as tf

import os

import cv2



data = []

for dirname, _, filenames in os.walk('/kaggle/input/fruits/fruits-360_dataset/fruits-360/Test'):

    if dirname != "/kaggle/input/fruits/fruits-360_dataset/fruits-360/Test":

        data.append(os.path.join(dirname, filenames[0]))

print(data[0:5])

print(len(data))

## DATASET LINK -> https://www.kaggle.com/moltean/fruits

## USING ONLY ONE IMAGE FROM EACH CATEGORY OF FRUITS

fruits = []

for i in data:

    newimage = cv2.imread(i, cv2.IMREAD_COLOR)

    RGB_image = cv2.cvtColor(newimage, cv2.COLOR_BGR2RGB)

    RGB_pixels = np.array(RGB_image)

    fruits.append(RGB_pixels)

#no preprocessing needed, all images are 100x100 images

fruits = np.array(fruits)

print(len(fruits))

print(fruits[0])
rn101 = tf.keras.applications.ResNet101(include_top=False, input_shape=(100,100,3))

rn152 = tf.keras.applications.ResNet152(include_top=False, input_shape=(100,100,3))
## 101

rn101.predict(fruits)
rn152.predict(fruits)