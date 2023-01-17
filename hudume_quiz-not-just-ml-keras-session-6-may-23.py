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
from keras.utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D,AveragePooling2D

from matplotlib.pyplot import plot

# visible=Input(shape=(8,8,1))

# conv1=Conv2D(32,kernel_size=4,activation="relu")(visible)

# pool1=MaxPooling2D(pool_size=(2,2))(conv1)

# conv2=Conv2D(16,kernel_size=4,activation="relu")(pool1)

# pool2=MaxPooling2D(pool_size=(2,2))(conv2)

# flat=Flatten()(pool2)

# hidden1=Dense(10,activation="relu")(flat)

# output=Dense(1,activation="sigmoid")(hidden1)

# model=Model(inputs=visible,outputs=output)

# print(model.summary())

# plot(model,to_file="CNN.png")

#Quiz code

input_layer=Input(shape=(8,8,1))

conv1=Conv2D(4,kernel_size=2,activation="relu")(input_layer)

pool1=MaxPooling2D(pool_size=(3,3),strides=(2,2))(conv1)

conv2=Conv2D(8,kernel_size=2,activation="relu")(pool1)

pool2=AveragePooling2D(pool_size=(2,2),strides=(2,2))(conv2)

flat=Flatten()(pool2)

hidden1=Dense(10,activation="relu")(flat)

output=Dense(1,activation="sigmoid")(hidden1)

model=Model(inputs=input_layer,outputs=output)

print(model.summary())
