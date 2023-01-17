# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

path_list = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path_list.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
path_list
train = pd.read_csv(path_list[2])
train.head()
train.shape
ytrain,xtrain = train.iloc[:,0],train.iloc[:,1:]
xtrain = np.array(xtrain)

ytrain = np.array(ytrain)
import numpy as np

from keras.utils import to_categorical

def preprocessing(x,y):

    x = np.reshape(x,(-1,28,28,1))

    x = x/255.0

    y = to_categorical(y)

    return x,y
xtrain,ytrain = preprocessing(xtrain,ytrain)

from keras.layers import *

from keras.models import Sequential

model = Sequential()

model.add(Conv2D(64,(3,3),activation = 'relu',input_shape = (28,28,1)))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation = 'relu'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.summary()
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['acc'])
model.fit(xtrain,ytrain,epochs = 20,validation_split=0.2)
test = pd.read_csv(path_list[1])
test = test.values
test.shape
test = np.reshape(test,(-1,28,28,1))

test =test/255.0
pred = model.predict_classes(test)
pred = list(pred)
result = pd.read_csv(path_list[0])
result.head()
result["Label"] = pred
result.head()
result.to_csv("/kaggle/working/result.csv")