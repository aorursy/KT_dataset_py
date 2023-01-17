import numpy as np
import pandas as pd

filename = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv"
df = pd.read_csv(filename)
df.head()
countries=['Italy','Brazil', 'Canada', 'Germany','US']
df2 = pd.DataFrame()
for c in countries:    
    df2[c] = df.loc[df['Country/Region']==c].iloc[0,4:]
    
df2.plot.line()
df2['US'].plot.line(logy=True,title='US on log scale')
from sklearn.preprocessing import MinMaxScaler

us = df2['US']
ustrain = us.head(77)
ustest = us.tail(10)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(us.values.reshape(-1,1))

ustrain = ustrain.values.reshape(-1,1)
ustrain = scaler.transform(ustrain)

ustest = ustest.values.reshape(-1,1)
ustest = scaler.transform(ustest)
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator

train_gen = TimeseriesGenerator(ustrain, ustrain, length=2, batch_size=10)     
test_gen = TimeseriesGenerator(ustest, ustest, length=2, batch_size=10)

model = Sequential()
model.add(LSTM(128,return_sequences=True, activation='relu',input_shape=(2,1)))
model.add(LSTM(128))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae')
model.fit_generator(train_gen, epochs=200, verbose=1)
ypred = model.predict_generator(test_gen)
import matplotlib.pyplot as plt

ustr = scaler.inverse_transform(ustrain).reshape((-1))
uste = scaler.inverse_transform(ustest).reshape((-1))
ypr = scaler.inverse_transform(ypred).reshape((-1))

plt.figure(figsize=(11,8))
plt.plot(ustr, marker='o')
plt.plot(range(77,87), uste, marker='o')
plt.plot(range(79,87), ypr, marker='o')
plt.legend(['Train','Test','Prediction'], loc='upper left')
plt.plot(range(77,87), uste, marker='o')
plt.plot(range(79,87), ypr, marker='o')
plt.legend(['Test','Prediction'], loc='upper left')
