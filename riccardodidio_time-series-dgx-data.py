# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
  print(tf.__version__)
except Exception:
  pass

import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras.layers import (Input, Activation, Dense, Flatten, Conv1D, 
                                     LSTM, Dropout, BatchNormalization)

import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        a =1

# Any results you write to the current directory are saved as output.
#URL = '/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/DGX_data.csv'
#URL = '/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/MOS_data.csv'
URL = '/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/INCY_data.csv'
data = pd.read_csv(URL)
data.head()
series = data['open'].to_numpy()
time   = data['date']
trace = go.Scatter(x=time, y=series,
                   mode='markers', name='markers')
layout = go.Layout(title='Time Series',
                   xaxis = {'title':'time'},
                   yaxis = {'title':'series'},
                   )
data = [trace]
fig = go.Figure(data=data, layout=layout)
fig.show()
split_time = int(0.7*len(series))
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size = 700
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
window_size = 64
batch_size = 256
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])
class model_1 (tf.keras.Model):
    
    def __init__(self):
       
        super(model_1, self).__init__()
        
        self.conv_1 = Conv1D(32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1])
        
        self.LSTM_1 = LSTM(64, return_sequences=True)
        self.LSTM_2 = LSTM(64, return_sequences=True) 
  
        self.flatten = tf.keras.layers.Flatten()
 
        self.dense_1 = tf.keras.layers.Dense(units=30, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=10, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(units=1)
        self.lamda_1 = tf.keras.layers.Lambda(lambda x: x * 400)
        
    def call(self, inputs):

        x = self.conv_1(inputs)
        x = self.LSTM_1(x)
        x = self.LSTM_2(x)
                
        x = self.flatten(x)
        x = self.dense_1(x)     
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.lamda_1(x)
        
        return x
model = model_1()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 60])
tf.keras.backend.clear_session()
train_set = windowed_dataset(x_train, window_size=60, batch_size=50, shuffle_buffer=shuffle_buffer_size)
model = tf.keras.models.Sequential([
          tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
          tf.keras.layers.LSTM(60, return_sequences=True),
          tf.keras.layers.LSTM(60, return_sequences=True),
          tf.keras.layers.Dense(30, activation="relu"),
          tf.keras.layers.Dense(10, activation="relu"),
          tf.keras.layers.Dense(1),
          tf.keras.layers.Lambda(lambda x: x * 50) 
])
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set,epochs=500)
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
true_value = go.Scatter(x=time_valid, y=x_valid,
                   mode='lines',
                    name='true_value',
                    marker = dict(
                       size = 12,
                       color = 'rgb(51,204,153)',
                   )
                       )
                    

predic_value = go.Scatter(x=time_valid, y=rnn_forecast,
                   mode='lines',
                    name='predic_value',
                    marker = dict(
                       size = 12,
                       color = 'rgb(200,204,20)',
                   )
                )
data = [true_value,predic_value]
layout = go.Layout(title='Compare true values and predict values')
fig = go.Figure(data=data, layout=layout)
fig.show()

