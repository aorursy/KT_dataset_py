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
import matplotlib.pyplot as plt

from keras.models import Sequential

from sklearn.model_selection import train_test_split

import csv as scv

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

import random

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

data=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
target=data['target']

train=data.drop(['id','target'],axis=1)
target=np.array(target)

target=target.reshape(250,1)

print(target.shape)
test_id=test['id']

test=test.drop('id',axis=1)
train=np.array(train)

test=np.array(test)

print(train.shape)
model=Sequential()
train=train.reshape(250,300)

print(train.shape)

x_train,x_valid,y_train,y_valid=train_test_split(train,target,test_size=0.2,random_state=0)

print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape)
model.add(Dense(64,input_dim=(300),activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(16,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(8,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))

model.compile(Adam(lr=0.001),loss="binary_crossentropy",metrics=["accuracy"])

model.summary()
history=model.fit(x_train,y_train,validation_split=0.1,epochs=100,verbose=1,shuffle=1)

prediction=model.predict_classes(x_valid)
prediction.shape

acc=accuracy_score(y_valid,prediction)

acc
prediction_test=model.predict_classes(test)
prediction_test.shape
x=prediction_test

x=x.reshape(19750,)
x.shape
submission={}

submission['id']=np.arange(250,20000)

submission['target']=x

submission=pd.DataFrame(submission)

submission.index=submission.index+1
submission.to_csv("submisision.csv", index=False)
submission