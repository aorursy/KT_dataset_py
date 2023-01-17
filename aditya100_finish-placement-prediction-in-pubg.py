import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_data = pd.read_csv('../input/train_V2.csv')
train_data.head()
train_data.describe()
train_data.shape
train_data = train_data.dropna()
train_data.isna().sum()
train_data.info()
_ = plt.figure(figsize=(30, 20))

p = sns.heatmap(train_data.corr(), annot=True)
Y_train = train_data['winPlacePerc']

X_train = train_data.drop(columns=['Id', 'groupId', 'matchId', 'winPlacePerc', 'matchType', 'DBNOs', 'headshotKills', 'matchDuration', 'maxPlace', 'numGroups', 'roadKills', 'vehicleDestroys', 'swimDistance'])



Y = Y_train.values

X = X_train.values
import tensorflow as tf

from tensorflow import keras

from keras import models

from keras import layers

from keras import Sequential

from keras.layers import Dense, Dropout, Input
model = Sequential()

model.add(Dense(80,input_dim=X_train.shape[1],activation='relu'))

model.add(Dense(160,activation='relu'))

model.add(Dense(320,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(160,activation='relu'))

model.add(Dense(80,activation='relu'))

model.add(Dense(40,activation='relu'))

model.add(Dense(20,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
history = model.fit(X, Y, epochs=50,

        batch_size=10000,

        validation_split=0.2,

        verbose=2)
plt.plot(history.history['mean_absolute_error'])

plt.plot(history.history['val_mean_absolute_error'])

plt.legend(['mean_absolute_error', 'val_mean_absolute_error'])

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['loss', 'val_loss'])
test_data = pd.read_csv('../input/test_V2.csv')
X_test = test_data.drop(columns=['Id', 'groupId', 'matchId', 'matchType', 'DBNOs', 'headshotKills', 'matchDuration', 'maxPlace', 'numGroups', 'roadKills', 'vehicleDestroys', 'swimDistance'])
predictions = model.predict(X_test).ravel()
predictions
sample_sub = pd.read_csv('../input/sample_submission_V2.csv')
sample_sub["winPlacePerc"] = predictions

sample_sub.head()
sample_sub.to_csv('sample_submission_v1.csv', index=False)