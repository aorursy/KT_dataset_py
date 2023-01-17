# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.metrics import confusion_matrix

import itertools

import seaborn as sns

from subprocess import check_output

import csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')
print(df)
df = df.apply(pd.to_numeric, errors = 'coerce')
df = df.fillna(df.mean())

df
z_train = Counter(df['target'])

z_train
sns.countplot(df['target'])
test = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')

test = test.apply(pd.to_numeric, errors = 'coerce')

test = test.fillna(test.mean())



print(test.shape)

test.head()
x_train = (df.iloc[:,1:].values).astype('float32')

y_train = (df.iloc[:,1].values).astype('int32')

x_test = test.values.astype('float32')

y_test = test.values.astype('int32')
x_test.shape
#%matplotlib inline

#plt.figure(figsize=(12,10))

#x, y = 10,4

#for i in range(40):

 #   plt.subplot(y, x, i+1)

  #  plt.imshow(x_train[i].reshape((1000,1000)),interpolation='nearest')

#plt.show()
from sklearn.preprocessing import MinMaxScaler
#normalise the data

scaler = MinMaxScaler()

scaler.fit_transform(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

x_train.shape
y_train = y_train.reshape(60000,1)

y_train.shape
print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
#X_train = x_train.reshape(x_train.shape[0], 57, 3, -1)

#X_test = x_test.reshape(x_test.shape[0], 57, 3,-1) 

from keras.datasets import mnist

from keras.preprocessing.image import load_img, array_to_img, ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline
#weights

#Biasis

#Input Layer > Weight > Hidden Layer > Output Layer

#cybenko's theorem says you only need one hidden layer for a good answer

#Use Relu

#Three Common Optimizers: SGD, RMSprop, Adam

#Loss function: mean_squared error, categorical_crossentropy, binary_crossentropy

model = Sequential()
model.add(Dense(214, activation='sigmoid', input_shape = (171,)))

model.add(Dense(214, activation='sigmoid'))

model.add(Dense(214, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=20)
plt.plot(history.history['accuracy'])

#plt.plot(history.history['validation_accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['loss'])
train_df = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')

test_df = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')
prediction = model.predict(y_test)
round_off_val = np.round_(prediction)

round_off_val.sum()
#from sklearn.ensemble import RandomForestClassifier



#rf_model = RandomForestClassifier()

#rf_model.fit(x_train, y_train)

#rf_model.score(x_train, y_train)

#print(rf_model.predict(x_test)[35])

#i  = 0

#while i < 16001:

 #   if(rf_model.predict(x_test)[i] == 1):

  #      rf_model.predict(x_test)[i] = 0

   # if(rf_model.predict(x_test)[i] == 0):

    #    rf_model.predict(x_test)[i] = 1

    #i+=1



#print(sum(rf_model.predict(x_test)))

            
#prediction['id'] = test['id']

#prediction.to_csv("submission.csv")

predictions = pd.DataFrame(round_off_val, columns=['target']).to_csv('submission.csv')