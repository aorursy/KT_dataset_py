# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import tensorflow.keras as keras

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/gold-price-data/gld_price_data.csv', index_col=0)



print(data.head())

#print(data.head())

#print(data.iloc[:,2])

print('in totale ci sono {} righe'.format(data.iloc[:,0].value_counts().sum()))



print(data.describe())

print('\n')

print(data.isnull().sum()) #no null values

#data['Date'] = data['Date'].apply(lambda x : int(x))
label = data.iloc[:,1]

data.drop(data.columns[1], axis=1, inplace=True)



print(label)



train_data,  test_data, train_label, test_label =  train_test_split(data, label, test_size=0.1)



print('train data has {} righe'.format(train_data.iloc[:,0].value_counts().sum()))

print('test data has {} righe'.format(train_label.size))



#train_data.drop(train_data.iloc[0,:], axis=1, inplace=True)



print('train_data: \n')

print(train_data.shape)

print('train_label: \n')

print(train_label.shape)



print('test_data: \n')

print(test_data.shape)

print('test_label: \n')

print(test_label.shape)
model = keras.models.Sequential()



model.add(keras.layers.Dense(4))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(128))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(1024))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(2048))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(1024))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(512))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(256))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(128))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(16))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(1))



model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
print(train_data.iloc[:,0].value_counts().sum())





#come gestisco le date???

#train_data.drop(train_data.columns[0], axis=1, inplace=True)



print(np.array(train_data).shape)

print(np.array(train_label).size)



print(train_data.head())

print(train_data.head())





model.fit(np.array(train_data), np.array(train_label), epochs=500)
model.evaluate(np.array(test_data), np.array(test_label))


for i in range(0,10):

    print(train_label[i])

    #print(np.array(train_data.iloc[i,:]).reshape(1,4))

    print(model.predict(np.array(train_data.iloc[i,:]).reshape(1,4)))
np.array(train_data).reshape(len(train_data),4)
import seaborn as sns



ax1 = sns.distplot(train_label, hist=False, color="r", label="Actual Value")

sns.distplot(model.predict(np.array(train_data).reshape(len(train_data),4)), hist=False, color="b", label="Fitted Values" , ax=ax1)