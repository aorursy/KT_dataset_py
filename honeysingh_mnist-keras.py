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
#import the libraries
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import np_utils
#first colomn is label and rest are pixels
df = pd.read_csv('../input/train.csv').values
X = df[:,1:].astype('float32')
Y = df[:,0]
#normalize the input
X/=255
#convert into one hot representation
Y = np_utils.to_categorical(Y)
Y.shape
#create the model
inputs = 28*28
output = 10
model = Sequential()
model.add(Dense(15,input_dim = inputs, activation='relu'))
model.add(Dense(output,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2)
#Fit the model
model.fit(X_train,Y_train,epochs=10)

