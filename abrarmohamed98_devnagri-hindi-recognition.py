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
from keras import layers

from keras.layers import Input,Dense,Activation,ZeroPadding2D,Flatten,Conv2D,MaxPooling2D,AveragePooling2D

from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D,BatchNormalization,Dropout

from keras.utils import np_utils,print_summary

from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

import keras.backend as k
data=pd.read_csv("../input/Project 1 data.csv")
data.head()
dataset=np.array(data)
np.random.shuffle(dataset)
x=dataset

y=dataset

x=x[:,0:1024]

y=y[:,1024]
x
y
y.shape
X_train = x[0:70000,:]
X_train.shape
X_train=X_train/255
Y_train = y[0:70000]
X_test=x[70000:,:]
X_test=X_test/255
Y_test=y[70000:]
Y_test.shape
Y_train=Y_train.T
Y_test=Y_test.T
X_train.shape
Y_train.shape
Y_test.shape
X_test.shape
train_y=np_utils.to_categorical(Y_train)
train_y.shape
test_y=np_utils.to_categorical(Y_test)
test_y.shape
X_train.shape
train_x=X_train.reshape(70000,32,32,1)
train_x.shape
test_x=X_test.reshape(2000,32,32,1)
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Conv2D(filters=64,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(37,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_x,train_y,batch_size=300,epochs=10,validation_split=0.1)
pred=model.predict(test_x)
pred=np.argmax(pred,axis=1)
pred
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, pred)
model.save('devnagri.h5')