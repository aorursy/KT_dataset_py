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
from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.models import Sequential

classifier=Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape=(64, 64, 3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,(3,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(activation='relu',units=128))

classifier.add(Dense(activation='sigmoid',units=1))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier2=classifier
from keras.preprocessing.image import ImageDataGenerator

import scipy.ndimage
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(

        '../input/drawings/spiral/training',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(

        '../input/drawings/spiral/testing',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')



train_generator2 = train_datagen.flow_from_directory(

        '../input/drawings/wave/training',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')
validation_generator2 = test_datagen.flow_from_directory(

        '../input/drawings/wave/testing',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')

classifier.fit_generator(

        train_generator,

        steps_per_epoch=8000,

        epochs=1,

        validation_data=validation_generator,

        validation_steps=2000)

classifier2.fit_generator(

        train_generator2,

        steps_per_epoch=8000,

        epochs=1,

        validation_data=validation_generator2,

        validation_steps=2000)