# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
TRAIN_DIR = '/kaggle/input/rock-paper-scissors-dataset/Rock-Paper-Scissors/train'
TEST_DIR = '/kaggle/input/rock-paper-scissors-dataset/Rock-Paper-Scissors/test'
VALID_DIR = '/kaggle/input/rock-paper-scissors-dataset/Rock-Paper-Scissors/validation'
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras import layers
datagen = ImageDataGenerator(rescale=1/255, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                            zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
train = datagen.flow_from_directory(TRAIN_DIR, target_size=(300,300), class_mode='categorical')
test = datagen.flow_from_directory(TEST_DIR, target_size=(300,300), class_mode='categorical')
import cv2
import matplotlib.pyplot as plt
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(300,300,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.4))

model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(216, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train, epochs=10)
model.evaluate_generator(test)
pred = model.predict_classes(test)
pred
