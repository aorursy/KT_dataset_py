from tensorflow import keras

from tensorflow.keras import layers

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



%matplotlib inline
df = pd.read_csv('../input/beijing-pm25-data-data-set/PRSA_data_2010.1.1-2014.12.31.csv')

df
df.info()
df['pm2.5'].isna().sum()
# drop the rows directly -> mess up the order

# first 24 rows have pm2.5 value that is NaN -> discard

# else: forward filling

df = df[24:].fillna(method='ffill')

df['pm2.5'].isna().sum()
import datetime



df['time'] = df.apply(lambda x : datetime.datetime(year=x['year'], month=x['month'], day=x['day'], hour=x['hour']), axis=1)

df.drop(columns=['year', 'month', 'day', 'hour', 'No'], inplace=True)

df = df.set_index('time')

df.head()
df['cbwd'].unique()
df = df.join(pd.get_dummies(df['cbwd'])) # one-hot encoding

del df['cbwd']

df.head()
df['pm2.5'][-1000:].plot()
df['TEMP'][-1000:].plot()
seq_len = 5*24 # observe the data for the past 5 days

delay = 1*24 # predict the PM2.5 value one day after



df_ = np.array([df.iloc[i : i + seq_len + delay].values for i in range(len(df) - seq_len - delay)])

df_.shape
np.random.shuffle(df_)

x = df_[:, :5*24, :]

y = df_[:, -1, 0]

x.shape, y.shape
split = int(y.shape[0]*0.8)

train_x = x[:split]

train_y = y[:split]

test_x = x[split:]

test_y = y[split:]



mean = train_x.mean(axis=0)

std = train_x.std(axis=0)

train_x = (train_x - mean) / std

test_x = (test_x - mean) / std # Use the mean & std of train. Since there's no way for us to know the future.
train_x.shape, test_x.shape
model = keras.Sequential()

model.add(layers.Flatten(input_shape=(120, 11)))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(1)) # Regression -> No Need for Activation



model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(train_x, train_y, batch_size=128, epochs=50, validation_data=(test_x, test_y))
plt.plot(history.epoch, history.history['mae'], c='m')

plt.plot(history.epoch, history.history['val_mae'], c='c')
model = keras.Sequential()

model.add(layers.LSTM(32, input_shape=(120, 11)))

model.add(layers.Dense(1))

model.summary()
model.compile(optimizer='adam', loss='mae')

history = model.fit(train_x, train_y, batch_size=128, epochs=150, validation_data=(test_x, test_y))
plt.plot(history.epoch, history.history['loss'], c='m')

plt.plot(history.epoch, history.history['val_loss'], c='c')
model = keras.Sequential()

model.add(layers.LSTM(32, input_shape=(120, 11), return_sequences=True)) 

model.add(layers.LSTM(32, return_sequences=True)) 

model.add(layers.LSTM(32)) 

model.add(layers.Dense(1))

model.summary()
lr_reduced = keras.callbacks.ReduceLROnPlateau('val_loss', patience=3, factor=0.5, min_lr=0.00001)
model.compile(optimizer='adam', loss='mae')

history = model.fit(train_x, train_y, batch_size=128, epochs=150, validation_data=(test_x, test_y), callbacks=[lr_reduced])
plt.plot(history.epoch, history.history['loss'], c='m')

plt.plot(history.epoch, history.history['val_loss'], c='c')
model.evaluate(test_x, test_y, verbose=0)
test_predict = model.predict(test_x)

test_x.shape, test_predict.shape
test_predict[:5]
test_data = df[-120:]

test_data = (test_data - mean)/std

test_data
test_data = np.expand_dims(test_data, axis=0)

test_data.shape
model.predict(test_data) # 2015.1.1 11pm pM2.5