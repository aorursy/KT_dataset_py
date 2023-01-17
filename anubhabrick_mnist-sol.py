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
import matplotlib.pyplot as plt

import keras

from keras.applications import vgg16

import numpy as np

import pandas as pd

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense,Flatten

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from sklearn import preprocessing

from keras.layers.convolutional import *
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")

sample=pd.read_csv("../input/digit-recognizer/sample_submission.csv")
mod = Sequential()

mod.add(Conv2D(filters=10, kernel_size=3 ,     

                  input_shape=(28,28,1),kernel_initializer= 'uniform',      

                  activation= 'relu'))

mod.add(Conv2D(filters=20, kernel_size=7 ,kernel_initializer= 'uniform',strides=(2,2),      

                  activation= 'relu'))

mod.add(MaxPooling2D(pool_size=(2,2)))

mod.add(Flatten())

mod.add(Dense(10,activation="softmax"))

mod.compile(Adam(lr=0.0001),metrics=["accuracy"],loss="categorical_crossentropy")

mod.summary()
name=train.columns.values[0]

ytrain=train[name]

xtrain=train.drop(name,axis=1)

xtrain=np.array(xtrain)

ytrain=keras.utils.to_categorical(ytrain)

test=np.array(test)
l=[]

for i in range(xtrain.shape[0]):

    l.append(np.reshape(xtrain[i],(28,28,1)))

xtrain=np.array(l)

xtrain.shape
l=[]

for i in range(test.shape[0]):

    l.append(np.reshape(test[i],(28,28,1)))

xtest=np.array(l)

xtest.shape
mod.fit(xtrain,ytrain,batch_size=10,epochs=10,verbose=2)
ypred=mod.predict(xtest,batch_size=10)
l=[]

for row in ypred:

    l.append(np.argmax(row)) 

Label=np.array(l)

sample["Label"]=Label


sample.to_csv("op_new.csv",index=False)

sample