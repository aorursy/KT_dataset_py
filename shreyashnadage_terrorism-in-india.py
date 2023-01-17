# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow import keras

from keras import Sequential

from keras.layers import Dense



import random



tf.set_random_seed(1); np.random.seed(1); random.seed(1)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

terror=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

# Any results you write to the current directory are saved as output.
terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)

terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

terror["Casualities"]=terror["Killed"]+terror["Wounded"]
terror_ind=terror[terror["Country"]=="India"]

city_null=terror_ind[terror_ind["latitude" and "longitude"].isnull()].city

len(terror_ind)


terror_ind=terror_ind.drop(index=city_null.index)

len(terror_ind)

terror_ind["Casualities"].fillna(0,inplace=True)

terror_ind.head(20)
terror_ind["Target_type"].unique()
hidden_units = (32,4)







model = Sequential()

model.add(Dense(30,input_shape=(5,), activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(30, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(

    # Technical note: when using embedding layers, I highly recommend using one of the optimizers

    # found  in tf.train: https://www.tensorflow.org/api_guides/python/train#Optimizers

    # Passing in a string like 'adam' or 'SGD' will load one of keras's optimizers (found under 

    # tf.keras.optimizers). They seem to be much slower on problems like this, because they

    # don't efficiently handle sparse gradient updates.

    tf.train.AdamOptimizer(0.005),

    loss='MSE',

    metrics=['MAE'],

)
X=terror_ind[["Year","Month","Day","latitude","longitude"]]

Y=terror_ind[["Casualities"]]

history = model.fit(

    X,

    Y,

    batch_size=500,

    epochs=30,

    verbose=1,

    validation_split=.05,

);
model.summary()
ip=np.array([2017,12,15,32,73])

ip = ip.reshape(-1,1)

ip.shape

ip

model.summary()
X.head()
Y.head()
t1=np.array(X.iloc[1].values)

print(t1)

print(Y.iloc[1].values)

t1=t1.reshape(1,-1)

test=model.predict(t1)
test[0]
op1=Y.iloc[0].values
op1
