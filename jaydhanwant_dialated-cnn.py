# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

pwd
os.chdir("/kaggle/input/cat-and-dog")
training_files = os.listdir("training_set/training_set/dogs")

training_files[1]

datadir = "training_set/training_set/dogs"

img_path = os.path.join(datadir, training_files[1])

import cv2

img_array = cv2.imread(img_path)
import matplotlib.pyplot as plt

print(img_array.shape)

plt.imshow(img_array, cmap = "gray")
# importing the data

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255, 

                                   shear_range = 0.2, 

                                   zoom_range = 0.2, 

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('training_set/training_set', 

                                                    target_size = (256, 256), 

                                                    batch_size = 128,

                                                   class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set/test_set',

                                                target_size = (256, 256),

                                                 batch_size = 128, 

                                                 class_mode = 'binary')

training_set[1][0][1].shape
training_set[3][0][1].shape
img_array = training_set[3][0][1]
import matplotlib.pyplot as plt

plt.imshow(img_array)
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

from keras.models import Sequential
model = Sequential()

model.add(Conv2D(32, kernel_size =(3,3), input_shape = (256,256,3), activation = "relu"))

model.add(MaxPool2D(2))

model.add(Conv2D(64, kernel_size =(3,3), input_shape = (256,256,3), activation = "relu"))

model.add(MaxPool2D(2))

model.add(Flatten())

model.add(Dense(32, activation = "relu"))

model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#model.summary()
model.fit(training_set,epochs = 50, validation_data = test_set)