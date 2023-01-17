import pandas as pd

import numpy as np



df = pd.read_csv("../input/apple-stock-data/AAPL.csv")



df['prev_day'] = df['Open'].shift(1)

df['change'] = (df['Open'] / df['prev_day']) - 1



df = df.dropna()



df
change = df['change']
x = []

y = []



for i in range(1, len(change) - 20):

    y.append(change[i])

    x.append(np.array(change[i+1:i+21]))



x = np.array(x).reshape(-1, 20, 1)

y = np.array(y)
from keras.models import Sequential

from keras.layers import LSTM



model = Sequential()

model.add(LSTM(1, input_shape=(20, 1)))



model.compile(optimizer='rmsprop', loss='mse')

model.fit(x, y, batch_size=32, epochs=10)
predictions = model.predict(x)
predictions = predictions.reshape(-1)
predictions
predictions = np.append(predictions, np.zeros(21))
predictions.shape
df['predictions'] = predictions
df['open_predict'] = df['prev_day']* (1+ df['predictions'])
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.cbook as cbook



months = mdates.MonthLocator()

monthsFmt = mdates.DateFormatter('%m')



ax = plt.gca()

ax.xaxis.set_major_locator(months)

ax.xaxis.set_major_formatter(monthsFmt)



dates = np.array(df["Date"]).astype(np.datetime64)



plt.plot(dates, df['Open'], label='open')

plt.plot(dates, df['open_predict'], label='open_predict')



plt.legend()



plt.show()
months = mdates.MonthLocator()

monthsFmt = mdates.DateFormatter('%m')



ax = plt.gca()

ax.xaxis.set_major_locator(months)

ax.xaxis.set_major_formatter(monthsFmt)



dates = np.array(df["Date"]).astype(np.datetime64)



plt.plot(dates[:20], df['Open'][:20], label='open')

plt.plot(dates[:20], df['open_predict'][:20], label='open_predict')



plt.legend()



plt.show()