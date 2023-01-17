# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/training_set"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import pandas as pd

import numpy as np
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten,Dropout
classifier=Sequential()
classifier.add(Conv2D(filters=16, kernel_size=(4,4) , strides=(2,2) ,padding='valid', input_shape=(128,128,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(1,1)))

classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters=32, kernel_size=(4,4) , strides=(1,1) ,padding='valid',activation='relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3),strides=(2,2)))

classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters=64, kernel_size=(3,3) , strides=(2,2) ,padding='valid',activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.2))
classifier.add(Conv2D(filters=128, kernel_size=(1,1) , strides=(1,1) ,padding='valid',activation='relu'))

classifier.add(MaxPooling2D(pool_size = (3,3),strides=(1,1)))

classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(units=512, activation='relu' ))

classifier.add(Dropout(0.2))
classifier.add(Dense(units=100, activation='relu' ))

classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation='sigmoid' ))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)
train_set = train_datagen.flow_from_directory(

        '../input/training_set/training_set',

        target_size=(128, 128),

        batch_size=16,

        class_mode='binary')
test_set = test_datagen.flow_from_directory(

        '../input/test_set/test_set',

        target_size=(128, 128),

        batch_size=16,

        class_mode='binary')
classifier.fit_generator(

        train_set,

        steps_per_epoch=8005,

        epochs=4,

        validation_data=test_set,

        validation_steps=2023)
classifier.save('./cat_and_dog_new.h5')