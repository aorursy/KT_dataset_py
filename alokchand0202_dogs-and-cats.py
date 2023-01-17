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
from keras.layers import Convolution2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3,  input_shape = (128,128,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(32, 3, 3, activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim=128, activation = "relu"))
classifier.add(Dense(output_dim=1, activation = "sigmoid"))
classifier.compile(optimizer = "adam", loss ="binary_crossentropy", metrics=["accuracy"])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../input/training_set/training_set/',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../input/test_set/test_set/',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')
classifier.fit_generator(training_set,
                   samples_per_epoch=8005,
                   nb_epoch=25,
                   validation_data=test_set,
                   nb_val_samples=2023)
