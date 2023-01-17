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


from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization

from keras.optimizers import Adam
import tensorflow as tf

import keras

print(tf.__version__)

print(keras.__version__)
model = Sequential()

model.add(Conv2D(64, (5,5),input_shape = (128,128,3), activation = 'relu')) ## initialize for all layers

#model.add(BatchNormalization(axis=-1))



model.add(Conv2D(128, (3,3), activation = 'relu'))

#model.add(BatchNormalization(axis=-1)) 

model.add(MaxPooling2D(pool_size = (2,2)))







model.add(Conv2D(256, (3,3), activation = 'relu'))

#model.add(BatchNormalization(axis=-1))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())

#model.add(BatchNormalization(axis=-1))



model.add(Dense(128, activation = 'relu'))

#model.add(Dropout(0.2))

#model.add(BatchNormalization(axis=-1))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=Adam(lr= 0.001),

              metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
# 1./255

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=False)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(

        '../input/training_set/training_set',

        target_size=(128, 128),

        batch_size=128,

        class_mode='binary')



test_set = test_datagen.flow_from_directory(

        '../input/test_set/test_set',

        target_size=(128,128),

        batch_size=128,

        class_mode='binary')



model.fit_generator(

        training_set,

        steps_per_epoch=8005//128,

        epochs=15,

        validation_data=test_set,

        validation_steps=2023//128)