import numpy as np

import pandas as pd

from sklearn import preprocessing
# load data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
#extract id

ids = test_data.values[:,0]
train_data.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'store_and_fwd_flag'], axis=1, inplace=True)

test_data.drop(['id', 'pickup_datetime', 'store_and_fwd_flag'], axis=1, inplace=True)
print("train: ", train_data.shape)

print("test: ", test_data.shape)
train_data.head()
test_data.head()
# MinMaxScaler

print('before scaling:', train_data.min().min(), train_data.max().max())



target = train_data['trip_duration']



for x in test_data:

    scaler = preprocessing.MinMaxScaler()

    scaler.fit(pd.concat([train_data[x], test_data[x]]).values.reshape(-1, 1))

    train_data[x] = scaler.transform(train_data[x].values.reshape(-1, 1))

    test_data[x] = scaler.transform(test_data[x].values.reshape(-1, 1)) 

    

train_data['target'] = target    



print('after scaling:', train_data.min().min(), train_data.max().max())
from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.regularizers import *
#Neural Network Architecture

model = Sequential()



#Input Layer

model.add(Dense(int(test_data.shape[1]), input_dim=test_data.shape[1], activation='relu', kernel_regularizer=l2(0.001)))



#layers

model.add(Dense(int(test_data.shape[1]), activation='relu', kernel_regularizer=l2(0.001)))

#model.add(Dense(int(test_data.shape[1]), activation='tanh', kernel_regularizer=l2(0.001)))

#model.add(Dense(int(test_data.shape[1]), activation='relu', kernel_regularizer=l2(0.001)))

#model.add(Dense(int(test_data.shape[1]), kernel_initializer='normal', activation='tanh', kernel_regularizer=l2(0.001)))

#model.add(Dense(int(test_data.shape[1] / 2), kernel_initializer='normal', activation='tanh', kernel_regularizer=l2(0.001)))

#model.add(Dense(int(test_data.shape[1] / 4), kernel_initializer='normal', activation='tanh', kernel_regularizer=l2(0.001)))

#model.add(Dense(int(test_data.shape[1] / 8), kernel_initializer='normal', activation='tanh', kernel_regularizer=l2(0.001)))



#Output later

model.add(Dense(1))
# Compile model

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'] ) 
# fit model

model.fit(train_data.values[:,:-1], train_data.values[:,-1], epochs=100, batch_size=5000)