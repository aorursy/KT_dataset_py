# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense

from keras.models import Sequential



import glob, os, random


import os

base_path='../input/garbage classification/Garbage classification'

img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

import matplotlib.pyplot as plt

path=plt.imread("../input/garbage classification/Garbage classification/trash/trash6.jpg")

plt.imshow(path)
train_datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    vertical_flip=True,

    validation_split=0.1

)



test_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.1

)



train_generator = train_datagen.flow_from_directory(

    base_path,

    target_size=(300, 300),

    batch_size=16,

    class_mode='categorical',

    subset='training',

    seed=0

)

validation_generator = test_datagen.flow_from_directory(

    base_path,

    target_size=(300, 300),

    batch_size=16,

    class_mode='categorical',

    subset='validation',

    seed=0

)



labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())



print(labels)
model = Sequential([

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),

    MaxPooling2D(pool_size=2),



    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),

    

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),

    

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),



    Flatten(),



    Dense(64, activation='relu'),



    Dense(6, activation='softmax')

])



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
model.fit_generator(train_generator, epochs=20, validation_data=validation_generator,steps_per_epoch=1000,validation_steps=100)