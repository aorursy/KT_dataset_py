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
#Importing required packages

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# checking a random picture from different categories

non_food = plt.imread('/kaggle/input/food5k-image-dataset/validation/non_food/433.jpg')

plt.imshow(non_food)

plt.title('Non food category image')

# checking a random picture from different categories

food = plt.imread('/kaggle/input/food5k-image-dataset/validation/food/270.jpg')

plt.imshow(food)

plt.title('Food category image')
# Creating a ImageGenerator

train_datagen = ImageDataGenerator(

                    rescale = 1./255)



train_generator = train_datagen.flow_from_directory(directory='/kaggle/input/food5k-image-dataset/training',

                                                   target_size=(128,128),

                                                   classes=['food','non_food'],

                                                   class_mode='binary')
# Creating Validation Generator

valid_datagen = ImageDataGenerator(

                    rescale = 1./255)



valid_generator = valid_datagen.flow_from_directory(directory='/kaggle/input/food5k-image-dataset/validation',

                                                   target_size=(128,128),

                                                   classes=['food','non_food'],

                                                   class_mode='binary')
# Creating Validation Generator

test_datagen = ImageDataGenerator(

                    rescale = 1./255)



test_generator = valid_datagen.flow_from_directory(directory='/kaggle/input/food5k-image-dataset/evaluation',

                                                   target_size=(128,128),

                                                   classes=['food','non_food'],

                                                   class_mode='binary')
#creating a Simple Convolution layer

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64,kernel_initializer='he_normal',kernel_size=(3,3),input_shape=(128,128,3),activation='relu'))

model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Dropout(0.3))



model.add(tf.keras.layers.Conv2D(128,kernel_initializer='he_normal',kernel_size=(3,3),activation='relu'))

model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Dropout(0.3))



model.add(tf.keras.layers.Conv2D(256,kernel_initializer='he_normal',kernel_size=(3,3),activation='relu'))

model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Dropout(0.5))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512,activation='relu'))

model.add(tf.keras.layers.Dense(1024,activation='relu'))

model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Y = model.fit_generator(train_generator, epochs=5,validation_data=valid_generator)
model.evaluate_generator(test_generator,steps=len(test_generator))