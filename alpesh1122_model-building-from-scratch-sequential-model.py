# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/flowers/flowers"))

# Any results you write to the current directory are saved as output.
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation,Dense, Flatten, Conv2D, Dropout
from tensorflow.python import keras

num_classes = 5
img_rows = 64
img_cols = 64
img_depth = 3
def get_model():
    model = Sequential()
    model.add(Conv2D(20, kernel_size=(3, 3),
                      activation='relu',
                      input_shape=(img_cols, img_rows, img_depth)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])    
    return model
    
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D,MaxPooling2D

image_dir = '../input/flowers/flowers'
data_generator = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2,
                                   rotation_range=30, 
                                   width_shift_range=0.1,
                                   height_shift_range=0.1, 
                                   shear_range=0.2, 
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode="nearest"
                                   )

train_generator = data_generator.flow_from_directory(
        image_dir,
        subset="training",
        target_size=(img_cols,img_rows),
        batch_size=32,
        class_mode='categorical')

#print(train_generator)
validation_generator = data_generator.flow_from_directory(
        image_dir,
        subset="validation",
        target_size=(img_cols,img_rows),
        class_mode='categorical')
#get new model and train with new data
my_new_model = get_model()
my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=4323,
        epochs=3,
        #validation_data=validation_generator,
        #validation_steps=1
        )

