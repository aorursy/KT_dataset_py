# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from keras.datasets import cifar10
from keras.applications import VGG16
from keras.layers import Dense, Input, Flatten, Conv2D
from keras.models import Model, Sequential
from keras.utils import to_categorical
(trainx, trainy), (testx, testy) = cifar10.load_data()
train_x = trainx / 255
test_x = testx / 255
train_y=to_categorical(trainy)
test_y=to_categorical(testy)
train_y

# i= Input(shape=(32, 32, 3))
# model = VGG16(input_shape=(32, 32, 3),
#               include_top=False)
model = Sequential()
for layer in VGG16(input_shape=(32, 32, 3), include_top=False, classes=10, weights='imagenet').layers:
    model.add(layer)
model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
settrainable=False
for layer in model.layers:
    if layer.name == 'block5_conv1':
        settrainable=True
    layer.trainable = settrainable
for layer in model.layers:
    print(layer.trainable)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_x, train_y,
                    validation_split=0.5,
                    epochs=30)
model.evaluate(test_x, test_y)
