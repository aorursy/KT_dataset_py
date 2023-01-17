# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import matplotlib.pyplot as plt

import os

#print(os.listdir("../input"))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
print(os.listdir("../input"))
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
x=train.drop(['label'],axis=1)

y=to_categorical(train['label'])
x=x.values.reshape(x.shape[0],28,28,1)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
#number_of_classes = ytest.shape[1]
import keras

from keras.models import Sequential

from keras.layers import Flatten,Dense

from keras.layers.convolutional import Conv2D,MaxPooling2D
model=Sequential()
model.add(Conv2D(64,activation='tanh',input_shape=(28,28,1),kernel_size=(2,2)))

model.add(Conv2D(32,activation='relu',kernel_size=(2,2)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=6,batch_size=1000,validation_data=(xtest,ytest))
x_test_recaled = (xtest.astype("float32") / 255)

scores = model.evaluate(x_test_recaled, ytest, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))