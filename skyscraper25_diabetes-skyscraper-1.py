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
# python3 p.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from keras import regularizers
import tensorflow as tf


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')

x = train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled)

y = test.values
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y)
test = pd.DataFrame(y_scaled)

# scaler = Normalizer().fit(x)
# normalizedX = scaler.transform(x)
# train = pd.DataFrame(normalizedX)



train_2, test_2 = train_test_split(train, test_size=0.2)

hidden_units=100
learning_rate=0.001
hidden_layer_act='tanh'
output_layer_act='sigmoid'
no_epochs=100

model = Sequential()

model.add(Dense(300, input_dim=8, activation=hidden_layer_act, kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dropout(0.2))
model.add(Dense(300, activation=hidden_layer_act, kernel_regularizer = regularizers.l2(0.01)))
model.add(Dense(300, activation=hidden_layer_act, kernel_regularizer = regularizers.l2(0.01)))
model.add(Dense(300, activation=hidden_layer_act, kernel_regularizer = regularizers.l2(0.01)))
model.add(Dense(1, activation=output_layer_act, kernel_regularizer = regularizers.l2(0.01)))

# sgd=optimizers.SGD(lr=learning_rate)
adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# rmsprop = optimizers.RMSprop(lr=learning_rate)
# adamax = optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy',optimizer=adamax, metrics=['acc'])

train_x=train_2.iloc[:,1:9]

train_y=train_2.iloc[:,9]

model.fit(train_x, train_y, epochs=no_epochs, batch_size= 128,  verbose=2)

val_loss, val_acc = model.evaluate(test_2.iloc[:,1:9], test_2.iloc[:,9])
print("Validation Loss : ", val_loss)
print("Validation Acc : ",val_acc)

test_x=test.iloc[:,1:9]
predictions = model.predict(test_x)

rounded = [int(round(x[0])) for x in predictions]

sample.diabetes = rounded
sample.to_csv('submission.csv',index = False)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')