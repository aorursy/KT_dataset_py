# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('../input/flowers-recognition/flowers/flowers/'))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import backend
script_dir = os.path.dirname(".")

script_dir
training_set_path = os.path.join(script_dir, '../input/flowers-recognition/flowers/flowers/')

test_set_path = os.path.join(script_dir, '../input/flowers-recognition/flowers/flowers/')
model = Sequential()
input_size = (256, 256)

batch_size = 100
model.add(Conv2D(64, kernel_size=3, input_shape=(256,256,3), activation='relu'))
model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.33)
training_set = train_datagen.flow_from_directory(training_set_path,

                                                 target_size=input_size,

                                                 batch_size=batch_size,

                                                 subset='training',

                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_set_path,

                                            target_size=input_size,

                                            batch_size=batch_size,

                                            subset='validation',

                                            class_mode='categorical')
training_set
model_info = model.fit_generator(training_set, epochs=10, validation_data=test_set)
model.summary()
pd.DataFrame(model_info.history).plot(figsize=(10,6), grid=True)