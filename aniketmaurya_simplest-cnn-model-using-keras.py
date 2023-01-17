# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/chest_xray/chest_xray/train"))

# Any results you write to the current directory are saved as output.
import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.applications import InceptionResNetV2, ResNet50
from keras.preprocessing.image import ImageDataGenerator
model = Sequential()
model.add(ResNet50(include_top = True, input_shape = (224, 224, 3)))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.layers[0].trainable = False
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
train_datagen = ImageDataGenerator(
        rescale=1./224,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False)

test_datagen = ImageDataGenerator(
        rescale=1./224,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(
        '../input/chest_xray/chest_xray/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../input/chest_xray/chest_xray/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        #steps_per_epoch=2000,
        epochs=2,
        validation_data=validation_generator,
        #validation_steps=800
)



