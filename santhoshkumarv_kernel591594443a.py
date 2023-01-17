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
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras

#from sklearn.cross_validation import train_test_split

from tensorflow.python.data import Dataset

from keras.utils import to_categorical

from keras import models

from keras import layers

#Read the data from csv file

df = pd.read_csv('../input/covtype.csv')

df.head()
#Select predictors

x = df[df.columns[:54]]

#Target variable 

y = df.Cover_Type

#Split data into train and test 

x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)

'''As y variable is multi class categorical variable, hence using softmax as activation function and sparse-categorical cross entropy as loss function.'''

model = keras.Sequential([

 keras.layers.Dense(64, activation=tf.nn.relu,                  

 input_shape=(x_train.shape[1],)),

 keras.layers.Dense(64, activation=tf.nn.relu),

 keras.layers.Dense(8, activation=  'softmax')

 ])



model.compile(optimizer=tf.train.AdamOptimizer(),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

history1 = model.fit(

 x_train, y_train,

 epochs= 26, batch_size = 60,

 validation_data = (x_test, y_test))
from sklearn import preprocessing

df = pd.read_csv('../input/covtype.csv')

x = df[df.columns[:55]]

y = df.Cover_Type

x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)

#Select numerical columns which needs to be normalized

train_norm = x_train[x_train.columns[0:10]]

test_norm = x_test[x_test.columns[0:10]]

# Normalize Training Data 

std_scale = preprocessing.StandardScaler().fit(train_norm)

x_train_norm = std_scale.transform(train_norm)

#Converting numpy array to dataframe

training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 

x_train.update(training_norm_col)

print (x_train.head())

# Normalize Testing Data by using mean and SD of training set

x_test_norm = std_scale.transform(test_norm)

testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 

x_test.update(testing_norm_col)

print (x_train.head())
model = keras.Sequential([

 keras.layers.Dense(64, activation=tf.nn.relu,                  

 input_shape=(x_train.shape[1],)),

 keras.layers.Dropout(0.5),

 keras.layers.Dense(64, activation=tf.nn.relu),

 keras.layers.Dropout(0.5),

 keras.layers.Dense(8, activation=  'softmax')

 ])



model.compile(optimizer=tf.train.AdamOptimizer (learning_rate=0.0001),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

history2 = model.fit(

 x_train, y_train,

 epochs= 5, batch_size = 60,

 validation_data = (x_test, y_test))