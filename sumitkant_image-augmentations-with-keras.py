import os

import cv2

import glob

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
SEED = 42 # will preserve the order of image displayed

ROWS = 3  # grid rows

COLS = 3  # grid size
files = glob.glob('../input/flower-color-images/flower_images/flower_images/*.png')
files_to_display = random.sample(files, ROWS*COLS) # sample of images from the folder



# reading, converting RGBA files to RGB and stacking images

X_train = np.vstack([cv2.cvtColor(plt.imread(f), cv2.COLOR_RGBA2RGB)  for f in files_to_display]) 

X_train = X_train.reshape(ROWS*COLS, 128, 128, 3)
def plot_images(datagen, X_train, multiplier=3):

    # plotting image

    f, axes = plt.subplots(ROWS, COLS, figsize = (COLS*multiplier, ROWS*multiplier))

    axes = axes.reshape(-1)

    

    # iterating over batch of image to display individual image

    batch = datagen.flow(X_train, batch_size=1, seed = SEED)

    for i in range(ROWS*COLS):

        axes[i].imshow(batch.next()[0])

        axes[i].axis('off')
from keras.preprocessing.image import ImageDataGenerator # importing the ImageDataGenerator API from keras



datagen = ImageDataGenerator() # intialize ImageDataGenerator

datagen.fit(X_train)           # fitting sample of images

plot_images(datagen, X_train)  # plotting the grid with augmentations
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

datagen.fit(X_train)

plot_images(datagen, X_train)
datagen = ImageDataGenerator(rotation_range = 90)

datagen.fit(X_train)

plot_images(datagen, X_train)
datagen = ImageDataGenerator(width_shift_range=0.2)

datagen.fit(X_train)

plot_images(datagen, X_train)
datagen = ImageDataGenerator(height_shift_range=0.2)

datagen.fit(X_train)

plot_images(datagen, X_train)
datagen = ImageDataGenerator(horizontal_flip=True)

datagen.fit(X_train)

plot_images(datagen, X_train)
datagen = ImageDataGenerator(vertical_flip=True)

datagen.fit(X_train)

plot_images(datagen, X_train)
datagen = ImageDataGenerator(zoom_range=0.3)

datagen.fit(X_train)

plot_images(datagen, X_train)
datagen = ImageDataGenerator(shear_range=45.0)

datagen.fit(X_train)

plot_images(datagen, X_train)
datagen = ImageDataGenerator(

    vertical_flip      = True,

    horizontal_flip    = True,

    rotation_range     = 90,

    width_shift_range  = 0.2,

    height_shift_range = 0.2,

    zoom_range         = 0.3

)



datagen.fit(X_train)

plot_images(datagen, X_train, multiplier=3)