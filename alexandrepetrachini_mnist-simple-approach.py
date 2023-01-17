# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.models import Sequential 

from keras.layers import Dense, Dropout

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix

from keras.datasets import mnist

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





        

data_train = pd.read_csv('../input/digit-recognizer/train.csv')

data_test = pd.read_csv('../input/digit-recognizer/test.csv')

        



y_train = data_train.iloc[:,0]

print(y_train.shape)

X_train = (data_train.iloc[:,1:].values).astype('float32')

print(X_train.shape)







X_test = (data_test.iloc[:,0:].values).astype('float32')





y_train = np_utils.to_categorical(y_train,10)

y_train =(np.array(y_train))









modelo = Sequential()

modelo.add(Dense(units = 64,activation = 'relu',input_dim = 784 ))

modelo.add(Dropout(0.2))

modelo.add(Dense(units = 64,activation = 'relu'))

modelo.add(Dropout(0.2))

modelo.add(Dense(units = 64,activation = 'relu'))

modelo.add(Dropout(0.2))

modelo.add(Dense(units = 10, activation = 'softmax'))



modelo.summary()

modelo.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

historico = modelo.fit(X_train, y_train, epochs = 20, validation_split = 0.2)



historico.history.keys()
