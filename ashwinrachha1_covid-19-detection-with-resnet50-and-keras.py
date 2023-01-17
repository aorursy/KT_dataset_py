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
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = '../input/chest-xray-covid19-pneumonia/Data/train'
test_dir = '../input/chest-xray-covid19-pneumonia/Data/test'

train_datagen = ImageDataGenerator(
    rescale = 1./255.0,
    height_shift_range = 0.2,
    width_shift_range = 0.2,
    rotation_range = 40,
    zoom_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

valid_datagen = ImageDataGenerator(rescale = 1./255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 100,
    class_mode = 'categorical',
    target_size = (300,300)
)

validation_generator = valid_datagen.flow_from_directory(
    test_dir,
    batch_size = 100,
    class_mode = 'categorical',
    target_size = (300,300)
)
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(300,300,3))
output = resnet.layers[-1].output
output = keras.layers.Flatten()(output)
resnet = Model(resnet.input,output)
for layer in resnet.layers:
    layer.trainable = False
resnet.summary()
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
input_shape = (300,300)
model = Sequential()
model.add(resnet)
model.add(Dense(512, activation = 'relu', input_dim = input_shape))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
             optimizer = optimizers.RMSprop(lr = 2e-5),
             metrics = ['accuracy'])
model.summary()

history = model.fit_generator(generator = train_generator,
                              validation_data = validation_generator,
                              epochs = 5,
                              verbose = 1)
