import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
catImages = os.listdir('../input/dog vs cat/dataset/training_set/cats')
print ("Number of Cat images - ",str(len(catImages)))

dogImages = os.listdir('../input/dog vs cat/dataset/training_set/dogs')
print ("Number of Dog images - ",str(len(dogImages)))
dogFilename = '../input/dog vs cat/dataset/training_set/dogs/'+dogImages[2]
dimage = Image.open(dogFilename)
dimage
catFilename = '../input/dog vs cat/dataset/training_set/cats/'+catImages[2]
CImage = Image.open(catFilename)
CImage
plt.figure(figsize=(10,10))
reqImage = cv2.imread(catFilename, cv2.IMREAD_GRAYSCALE)
print (reqImage.shape)
plt.imshow(reqImage)
plt.figure(figsize=(10,10))

reqImage = cv2.imread(dogFilename, cv2.IMREAD_COLOR)           # rgb
alpha_img = cv2.imread(dogFilename, cv2.IMREAD_UNCHANGED) # rgba
gray_img = cv2.imread(dogFilename, cv2.IMREAD_GRAYSCALE)  # grayscale
print (reqImage.shape)
print (gray_img.shape)
plt.imshow(gray_img)

import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
# Building CNN
classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3,input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
