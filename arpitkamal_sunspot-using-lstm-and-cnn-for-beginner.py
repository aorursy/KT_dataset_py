# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

print(tf.__version__)
#for plotting charts

import numpy as np

import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None):

    plt.plot(time[start:end], series[start:end], format)

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.grid(True)
# using pandas to import data instead of CSV



data = pd.read_csv('/kaggle/input/sunspots/Sunspots.csv')

data.head()
series = data['Monthly Mean Total Sunspot Number'].to_numpy()

series[:10]
time = data['Unnamed: 0'].to_numpy()

time[:10]
plt.figure(figsize=(10, 6))

plot_series(time, series)
#splitting the data into training and validation set

split_time = 2500

time_train = time[:split_time]

x_train = series[:split_time]

time_valid = time[split_time:]

x_valid = series[split_time:]





window_size = 30

batch_size = 32

shuffle_buffer_size = 1000

#function for turning the data into a Window data set.

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
tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)

window_size = 64

batch_size = 256

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

print(train_set)

print(x_train.shape)

# 1d convolutional neural network

# LSTM

# DNN

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



lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])





plt.semilogx(history.history["lr"], history.history["loss"])

plt.axis([1e-8, 1e-4, 0, 60])
tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)

train_set = windowed_dataset(x_train, window_size=60, batch_size=200, shuffle_buffer=shuffle_buffer_size)

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

          tf.keras.layers.Lambda(lambda x: x * 400) 

])





optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

history = model.fit(train_set,epochs=500)
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)

rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, rnn_forecast)
tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
print(rnn_forecast)
import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

mae=history.history['mae']

loss=history.history['loss']



epochs=range(len(loss)) # Get number of epochs



#------------------------------------------------

# Plot MAE and Loss

#------------------------------------------------

plt.plot(epochs, mae, 'r')

plt.plot(epochs, loss, 'b')

plt.title('MAE and Loss')

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend(["MAE", "Loss"])



plt.figure()



epochs_zoom = epochs[200:]

mae_zoom = mae[200:]

loss_zoom = loss[200:]



#------------------------------------------------

# Plot Zoomed MAE and Loss

#------------------------------------------------

plt.plot(epochs_zoom, mae_zoom, 'r')

plt.plot(epochs_zoom, loss_zoom, 'b')

plt.title('MAE and Loss')

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend(["MAE", "Loss"])



plt.figure()