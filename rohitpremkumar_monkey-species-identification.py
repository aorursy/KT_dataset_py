# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import cv2
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
training_dir = '../input/10-monkey-species/training/training'
test_dir = '../input/10-monkey-species/validation/validation'
# Any results you write to the current directory are saved as output.
img_shape = 300
num_classes = 10

model = Sequential()
model.add(Conv2D(30, kernel_size=(4, 4),
                 strides=2,
                 activation='relu',
                 input_shape=(img_shape, img_shape, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(30, kernel_size=(4, 4), strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(30, kernel_size=(4, 4), strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(img_shape, img_shape),
        color_mode='rgb',
        batch_size=24,
        class_mode='categorical',
        shuffle=True)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_shape, img_shape),
        color_mode='rgb',
        batch_size=1,
        shuffle=False,
        class_mode='categorical')

pd.read_csv('../input/10-monkey-species/monkey_labels.txt')
step_size_train = train_generator.n//train_generator.batch_size
step_size_test = test_generator.n//test_generator.batch_size

stopper = EarlyStopping(monitor='val_acc', patience=10)
callbacks = [stopper]
model.fit_generator(
    train_generator,
    steps_per_epoch=step_size_train,
    validation_data=test_generator,
    validation_steps=step_size_test,
    epochs = 1, 
    callbacks = callbacks
)

