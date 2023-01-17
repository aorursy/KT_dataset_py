# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras as k

from keras.models import Sequential

from keras.layers import Dense, Activation, BatchNormalization, Dropout

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from keras.losses import mean_absolute_error,mean_squared_error

from keras import backend as K

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

K.tensorflow_backend._get_available_gpus()



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train_V2.csv")

df_test = pd.read_csv("../input/test_V2.csv")

out = pd.read_csv("../input/sample_submission_V2.csv")
df.head()
df[df['winPlacePerc'].isnull()]

df.drop(2744604, inplace=True)

df[df['winPlacePerc'].isnull()]
df['movement'] = df['swimDistance'] + df['walkDistance'] + df['rideDistance']

df.drop(['swimDistance','walkDistance','rideDistance'], inplace=True, axis=1)

column_names = ['assists', 'boosts', 'damageDealt', 'DBNOs',

       'headshotKills', 'heals', 'killPlace', 'kills',

       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',

       'numGroups', 'revives', 'roadKills', 'teamKills',

       'vehicleDestroys', 'weaponsAcquired', 'winPoints','movement']

x = df[column_names]

y = df['winPlacePerc']

x.shape
for i in x:

    de = np.max(x[i]) - np.min(x[i])

    x[i] -= np.min(x[i])

    x[i] = x[i] / (de)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 705023)
m = Sequential()

m.add(Dense(256, input_dim = 20, init = 'uniform'))

m.add(BatchNormalization())

m.add(Activation('relu'))

m.add(Dropout(0.5))



m.add(Dense(256, init = 'uniform'))

m.add(BatchNormalization())

m.add(Activation('relu'))

m.add(Dropout(0.5))



m.add(Dense(256, init = 'uniform'))

m.add(BatchNormalization())

m.add(Activation('relu'))

m.add(Dropout(0.5))



m.add(Dense(256, init = 'uniform'))

m.add(BatchNormalization())

m.add(Activation('relu'))

m.add(Dropout(0.5))



m.add(Dense(256, init = 'uniform'))

m.add(BatchNormalization())

m.add(Activation('relu'))



m.add(Dense(1, init = 'uniform'))

m.add(BatchNormalization())

m.add(Activation('sigmoid'))

m.summary()
m.compile(loss='mae', optimizer=Adam(lr=0.05 ), metrics=['mae'])

es = EarlyStopping(monitor='val_loss',patience=8,baseline=None, restore_best_weights=False)
hist = m.fit(x_train,y_train,batch_size=500, validation_split=0.1,epochs=70)
df_test['movement'] = df_test['swimDistance'] + df_test['walkDistance'] + df_test['rideDistance']

df_test.drop(['swimDistance','walkDistance','rideDistance'], inplace=True, axis=1)

x = df_test[column_names]



for i in x:

    de = np.max(x[i]) - np.min(x[i])

    x[i] -= np.min(x[i])

    x[i] = x[i] / (de)





pred = m.predict(x)

out['winPlacePerc'] = pred

out.to_csv("myans.csv",index=False)