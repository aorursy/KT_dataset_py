# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import tensorflow as tf





# Any results you write to the current directory are saved as output.
DF1 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126 - SUMMARY.csv")

DF1 = DF1.drop(["Province/State"],axis=1)

#for i in DF1['Date last updated'].to_list():

    #DF1['Date last updated'] = i[0:8]

DF1['Date last updated'] = pd.to_datetime(DF1['Date last updated'], format='%m/%d/%y',exact = False)

DF1['month'] =  DF1['Date last updated'].apply(lambda x: x.month)/10

DF1['day'] =  DF1['Date last updated'].apply(lambda x: x.day)/100

DF1 =  DF1.drop(['Date last updated'],axis = 1)

DF1['Confirmed'] = DF1['Confirmed'].fillna(0)/10

DF1 = DF1.drop(["Suspected"],axis=1)

DF1 = DF1.drop(["Recovered"],axis=1)

DF1 = DF1.drop(["Deaths"],axis=1)

countries = []

counter = 0

DF1 = pd.get_dummies(DF1,columns=['Country'],drop_first=True)

DF1.to_csv("preprocessed1.csv")
y = DF1['Confirmed'].to_numpy()

X = DF1.drop(['Confirmed'],axis=1).to_numpy()

x_train,x_val,y_train,y_val = train_test_split(X, y, test_size=0.3, random_state=42)
DF1


simple_lstm_model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(21, input_shape=(22,),activation='sigmoid'),

    tf.keras.layers.Dense(14,activation='sigmoid'),

    tf.keras.layers.Dense(14,activation='sigmoid'),

    tf.keras.layers.Dense(1,activation='sigmoid'),

])

sgd = tf.keras.optimizers.Adam(lr=0.003)

simple_lstm_model.compile(optimizer=sgd, loss='mae')



EPOCHS = 300



simple_lstm_model.fit(x_train,y_train, epochs=EPOCHS,validation_data=(x_val, y_val))

simple_lstm_model.predict(x_train)
simple_lstm_model.save('GeorgeIsAGenius.h5')
x_train[1]