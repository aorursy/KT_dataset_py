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
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


train_x=train.iloc[:,1:]
train_y=train.iloc[:,:1]

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical

train_x = train_x.astype('float32')

train_x /= 255

y = to_categorical(train_y)
print(train_x.head())
print(train_y.head())
model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0.7))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.7))
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',metrics=["accuracy"],optimizer='rmsprop')


model.fit(train_x,y,batch_size=128,epochs=15,verbose=1,validation_split=1/7)
