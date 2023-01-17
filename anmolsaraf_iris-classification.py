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
import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from keras.models import Sequential

from keras.layers import Dense
df_iris=pd.read_csv('../input/iris/Iris.csv')

df_iris.head()
y=df_iris["Species"]

x=df_iris.values

x=x[:,1:-1]

min_max_scaler = preprocessing.MinMaxScaler()

x = min_max_scaler.fit_transform(x)

print(x[:5,:])

x_train,x_test,y_train,y_test=train_test_split(x,y.values,test_size=0.2)
y_train=y_train.reshape(len(y_train),1)

y_test=y_test.reshape(len(y_test),1)

le=preprocessing.OrdinalEncoder()

le.fit(y_train)

y_train_num =le.transform(y_train)

y_test_num =le.transform(y_test)

y_train_num[:10]
test_model=Sequential()

test_model.add(Dense(8,input_shape= (4,),activation='relu'))

test_model.add(Dense(12,activation='relu'))

test_model.add(Dense(10,activation='relu'))

test_model.add(Dense(3,activation='softmax'))

test_model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

test_model.fit(x_train, y_train_num, batch_size= 20, epochs=100)
test_model.evaluate(x_test, y_test_num)
a= np.argmax(test_model.predict(x_test[2:3]))

print("Predicted value: %s, Actual VAlue: %s" %(a, y_test_num[2]))