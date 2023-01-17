# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import tensorflow as tf

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
f_x=pd.read_csv('../input/mushrooms.csv')

x=f_x.iloc[:,1:]

y=f_x.iloc[:,0]

# converting labels to integers

y.replace('p',0,inplace=True)

y.replace('e',1,inplace=True)



x=pd.get_dummies(x)



x=x.values.astype('int32')

y=y.values.astype('int32')



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=keras.Sequential([keras.layers.Dense(50,activation=tf.nn.relu),

                        keras.layers.Dense(2,activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)
y_hat=model.predict(x_test)

perf=model.evaluate(x_test,y_test)

print(perf)
ind=0

print(y_hat[ind])

print(y_test[ind])