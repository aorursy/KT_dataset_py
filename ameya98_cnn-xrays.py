# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/chest_xray/chest_xray"))



# Any results you write to the current directory are saved as output.
import tensorflow.keras

import tensorflow as tf

import pandas as pd

import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten,Dense,MaxPooling2D,Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)
train_set = train_datagen.flow_from_directory(

        '../input/chest_xray/chest_xray/train',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')
test_set = test_datagen.flow_from_directory(

        '../input/chest_xray/chest_xray/test',

        target_size=(64, 64),

        batch_size=32

    ,

        class_mode='binary')

classifier=Sequential()
classifier.add(Conv2D(filters=64, kernel_size=(3,3) , strides=(1,1) ,padding='valid', input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(1,1)))
classifier.add(Conv2D(filters=128, kernel_size=(1,1) , strides=(1,1) ,padding='valid',activation='relu'))
classifier.add(MaxPooling2D(pool_size = (4, 4),strides=(2,2)))
classifier.add(Conv2D(filters=256, kernel_size=(3,3) , strides=(2,2) ,padding='valid', input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3),strides=(1,1)))
classifier.add(Flatten())
classifier.add(Dense(units=256, activation='relu' ))

classifier.add(Dense(units=128, activation='relu' ))

classifier.add(Dense(units=1, activation='sigmoid' ))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.summary()
classifier.fit_generator(

        train_set,

        steps_per_epoch=5216,

        epochs=1,

        validation_data=test_set,

        validation_steps=624)
classifier.save('medical.h5')