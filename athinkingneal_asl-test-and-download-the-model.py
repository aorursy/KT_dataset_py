# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import h5py

#import h5py to export the model into a .h5 file
# Imports for Deep Learning

from keras.layers import Conv2D, Dense, Dropout, Flatten

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

import keras



# ensure consistency across runs

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)



# Imports to view data

import cv2

from glob import glob

from matplotlib import pyplot as plt

from numpy import floor

import random



def plot_three_samples(letter):

    print("Samples images for letter " + letter)

    base_path = '../input/asl_alphabet_train/asl_alphabet_train/'

    img_path = base_path + letter + '/**'

    path_contents = glob(img_path)

    

    plt.figure(figsize=(16,16))

    imgs = random.sample(path_contents, 3)

    plt.subplot(131)

    plt.imshow(cv2.imread(imgs[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(imgs[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(imgs[2]))

    return



plot_three_samples('A')
data_dir = "../input/asl_alphabet_train/asl_alphabet_train"

target_size = (64, 64)

target_dims = (64, 64, 3) # add channel for RGB

n_classes = 29

val_frac = 0.1

batch_size = 64



data_augmentor = ImageDataGenerator(samplewise_center=True, 

                                    samplewise_std_normalization=True, 

                                    validation_split=val_frac)



train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")

val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")
my_model = Sequential()

my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))

my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))

my_model.add(Dropout(0.5))

my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))

my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))

my_model.add(Dropout(0.5))

my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))

my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))

my_model.add(Flatten())

my_model.add(Dropout(0.5))

my_model.add(Dense(512, activation='relu'))

my_model.add(Dense(n_classes, activation='softmax'))



my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
steps_per_epoch=len(train_generator) 

validation_steps=len(val_generator) 

my_model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=5, validation_data=val_generator)


my_model.save('my_model.h5')


