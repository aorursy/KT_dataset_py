# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
batch_size = 32
num_classes = 5
input_size =  (64, 64)
training_path = "../input/flowers/flowers"
test_path = "../input/flowers/flowers"
train_data_generator = ImageDataGenerator(rescale= 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data_generator = ImageDataGenerator(rescale=1. / 255, validation_split = 0.2)

training_set = train_data_generator.flow_from_directory(training_path,target_size=input_size,batch_size=batch_size,
                                                        subset="training",
                                                        class_mode='categorical')
                                                        
test_set = test_data_generator.flow_from_directory(test_path,target_size=input_size,batch_size=batch_size,subset="validation",class_mode='categorical')
model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), input_shape = (64, 64, 3), activation = 'relu'))
model.add(Conv2D(48, kernel_size = (5, 5), activation = 'relu'))
model.add(Flatten())
model.add(Dense(125, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
model_info = model.fit_generator(training_set, steps_per_epoch = 4323/batch_size, epochs = 50, validation_data = test_set, validation_steps = 100/batch_size, workers = 1)

