# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.keras as keras

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/pmsm_temperature_data.csv')



print(data.describe())



#print(data.isnull().sum()) #non ci sono valori nulli



#prendo la label

label = data['pm']



#prendo i dati

data.drop('pm', axis=1, inplace=True)
print('data number of rows is {}'.format(data.iloc[:,0].value_counts().sum()))

print('label number of rows is {}'.format(label.value_counts().sum()))

data.pop('profile_id')

print(plt.matshow(data.corr()))





print(data.head())
#divido in test set



train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1)





print('train_data.size is {}'.format(train_data.iloc[:,1].value_counts().sum()))

print('train_label.size is {}'.format(train_label.value_counts().sum()))



print('test_data.size is {}'.format(test_data.iloc[:,1].value_counts().sum()))

print('test_label.size is {}'.format(test_label.value_counts().sum()))





model = keras.models.Sequential()



model.add(keras.layers.Dense(32, input_shape=([len(train_data.keys())])))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(256))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(512))

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



model.add(keras.layers.Dense(64))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(32))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(1))



model.summary()



model.compile(optimizer='adam', loss='mean_absolute_error' , metrics=['mean_absolute_error'])

history = model.fit(train_data, train_label, epochs=10)

# NOT OK ..... history = model.evaluate(test_data, test_label)
acc = history.history['mean_absolute_error']

epochs_=range(0,10)  



plt.plot(epochs_, acc, label='mean_absolute_error')

plt.xlabel('no of epochs')

plt.ylabel('mean_absolute_error')



#acc_val = history.history['mean_absolute_error']

#plt.plot(epochs_, acc_val, label="validation mean_squared_error")



plt.title("no of epochs vs accuracy")

plt.legend()



import seaborn as sns



ax1 = sns.distplot(train_label, hist=False, color="r", label="Actual Value")

sns.distplot(model.predict(train_data), hist=False, color="b", label="Fitted Values" , ax=ax1)
model.evaluate(test_data, test_label)



for i in range(0,20):

    print('target values is {} and the diff with the prediction is {}'.format(train_label.iloc[[i]],model.predict(train_data.iloc[[i]]) ))

    #print('and prediction is {}'.format(model.predict(train_data.iloc[[i]])))