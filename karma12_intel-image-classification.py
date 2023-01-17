# Import Required Libraries

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

from tensorflow.keras import Model

from tensorflow.keras.utils import plot_model

from tensorflow.keras.applications.inception_v3 import InceptionV3
print(os.listdir('../input/intel-image-classification'))
train_dir = os.path.join('../input/intel-image-classification/seg_train/seg_train')

test_dir = os.path.join('../input/intel-image-classification/seg_test/seg_test')
print(os.listdir(train_dir))
print(os.listdir(test_dir))
datagen = ImageDataGenerator(

    rescale = 1./255,

    horizontal_flip = True,

    vertical_flip = True,

    shear_range = 0.2,

    zoom_range = 0.2,

    rotation_range = 40,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    #fill_mode = 'nearest',

    validation_split=0.2

)
train_data_gen = datagen.flow_from_directory(os.path.join(train_dir),

                                              target_size=(150, 150),

                                             batch_size = 32,

                                             class_mode='categorical',

                                             subset='training')

test_data_gen = datagen.flow_from_directory(os.path.join(test_dir),

                                              target_size=(150, 150),

                                             batch_size = 32,

                                             class_mode='categorical',

                                             subset='validation')
plt.imshow(train_data_gen[0][0][10]);
import tensorflow as tf

model = Sequential([

    Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 

    MaxPool2D(2,2),

    Conv2D(32, (3, 3), activation = 'relu'),

    MaxPool2D(2,2),

    Conv2D(64, (3, 3), activation = 'relu'),

    MaxPool2D(2,2),

    Dropout(0.1),

    Conv2D(64, (3, 3), activation = 'relu'),

    MaxPool2D(2,2),

    Dropout(0.2),

    Conv2D(128, (3, 3), activation = 'relu'),

    MaxPool2D(2,2),

    Dropout(0.1),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(6, activation='softmax')

])
model.summary()
plot_model(model)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics= ['accuracy'])
history = model.fit(train_data_gen,validation_data = test_data_gen,

                    epochs=100,steps_per_epoch = 20,verbose=2,

                    validation_steps=3)