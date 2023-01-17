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
print(os.listdir("../input/training_set/Training_set"))
import keras 

from keras.preprocessing import image

from IPython.display import Image

from keras.preprocessing.image import ImageDataGenerator

import cv2
datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True)

batch_size = 100
path = "../input/training_set/Training_set"

train_generator = datagen.flow_from_directory(path,batch_size = batch_size,target_size =(150,150),class_mode = 'binary')
from keras import layers

from keras import models
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32,kernel_size = (2,2),activation = 'relu',padding = "same",strides = (2,2),input_shape = (150,150,3)))

model.add(keras.layers.Conv2D(64,kernel_size = (4,4),activation = 'relu',padding = "same",strides = (2,2)))

model.add(keras.layers.Conv2D(64,kernel_size = (2,2),activation = 'relu',strides = (2,2)))

model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(128,kernel_size = (4,4),activation = 'relu',padding = "same",strides = (2,2)))

model.add(keras.layers.Conv2D(128,kernel_size = (2,2),activation = 'relu',strides = (2,2)))

model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(256,activation = 'relu'))

model.add(keras.layers.Dense(1,activation = "sigmoid"))
model.compile(loss = "binary_crossentropy",optimizer = 'Adam',metrics = ['accuracy'])
model.fit_generator(train_generator,steps_per_epoch = 50, epochs = 20)
path = "../input/test/test"

test_generator = datagen.flow_from_directory(path,target_size = (150,150),class_mode = 'binary',batch_size = batch_size)
model.evaluate_generator(test_generator,verbose = 1,steps = len(test_generator))
model.metrics_names