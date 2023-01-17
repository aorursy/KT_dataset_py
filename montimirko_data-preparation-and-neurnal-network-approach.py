# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.keras as keras



from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/renfe.csv', index_col = 0)
print(data.head())

data.describe()

data.info()

#verify null values:

print(data.isnull().sum())
mean_price = data['price'].mean()

print(mean_price)
data['price'].fillna(mean_price, inplace=True)

print('now we have 0 null price')

print(data.isnull().sum()) 

print('now we will drop the null values')

data.dropna(inplace=True) # droppo

print(data.isnull().sum())

print(data.head())
data.drop('insert_date',axis=1,inplace=True)

data.drop(['start_date','end_date'],axis=1,inplace=True)

print(data.head())
lab_en = LabelEncoder()

data.iloc[:,0] = lab_en.fit_transform(data.iloc[:,0])

data.iloc[:,1] = lab_en.fit_transform(data.iloc[:,1])

data.iloc[:,2] = lab_en.fit_transform(data.iloc[:,2])



data.iloc[:,4] = lab_en.fit_transform(data.iloc[:,4])

data.iloc[:,5] = lab_en.fit_transform(data.iloc[:,5])



#data normalization

data.iloc[:,0] = data.iloc[:,0]/(data.iloc[:,0].max())

data.iloc[:,1] = data.iloc[:,1]/(data.iloc[:,1].max())

data.iloc[:,2] = data.iloc[:,2]/(data.iloc[:,2].max())

data.iloc[:,4] = data.iloc[:,4]/(data.iloc[:,4].max())

data.iloc[:,5] = data.iloc[:,5]/(data.iloc[:,5].max())



print(data.head())
# ora creo il train/test set e le labels

X = data.iloc[:,[0,1,2,3,5]].values

Y = data.iloc[:,4].values



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)



print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)



print(X_train[1])

print(Y_train[1])
myModel = keras.models.Sequential([

    #keras.layers.Input((X_train.shape[1],)),

    keras.layers.Flatten(input_shape=(X_train.shape[1],)), #non va bene si aspetta 3d

    keras.layers.Dense(32,kernel_initializer='he_normal', activation = tf.nn.relu),

    keras.layers.Dense(32,kernel_initializer='he_normal', activation=tf.nn.relu),

    keras.layers.Dense(5, kernel_initializer='he_normal', activation=tf.nn.relu),

    keras.layers.Dense(1, activation=tf.nn.tanh)

    ])



myModel.compile(loss=tf.keras.losses.mean_squared_error, optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), 

                metrics=[ 'mean_squared_error'] )



myModel.summary()



#let's train our algorithm

myModel.fit(X_train, Y_train , epochs=3)





#evaluate:

myModel.evaluate(X_test, Y_test)