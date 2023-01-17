# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None):

    plt.plot(time[start:end], series[start:end], format)

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.grid(True)
import pandas as pd

sunspots = pd.read_csv("/kaggle/input/sunspots/Sunspots.csv")

sunspots
series = np.array(sunspots['Monthly Mean Total Sunspot Number'])

time = np.array(sunspots.index)



plt.figure(figsize = (10,6))

plot_series(time, series)
split_time = 3000

time_train = time[:split_time]

x_train = series[:split_time]

time_valid = time[split_time:]

x_valid = series[split_time:]



window_size = 30

batch_size = 32

shuffle_buffer_size = 1000
def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):

    series = tf.expand_dims(series, axis = -1)

    dataset = tf.data.Dataset.from_tensor_slices(series)

    dataset = dataset.window(window_size + 1, shift = 1, drop_remainder = True)

    dataset = dataset.flat_map(lambda window : window.batch(window_size + 1))

    dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.map(lambda window : (window[:-1], window[-1:]))

    return dataset.batch(batch_size).prefetch(1)
def model_forecast(model, series, window_size):

    dataset = tf.data.Dataset.from_tensor_slices(series)

    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    dataset = dataset.batch(32).prefetch(1)

    forecast = model.predict(dataset)

    return forecast
tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)

window_size = 64

batch_size = 256

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

print(train_set)

print(x_train.shape)



model = tf.keras.models.Sequential([

  tf.keras.layers.Conv1D(filters=60, kernel_size=5, strides=1, padding="causal",activation="relu", input_shape=[None, 1]),

  tf.keras.layers.LSTM(60, return_sequences=True),

  tf.keras.layers.BatchNormalization(),  

  tf.keras.layers.LSTM(60),

  tf.keras.layers.BatchNormalization(),  

  tf.keras.layers.Dense(30, activation="relu"),

  tf.keras.layers.BatchNormalization(),  

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

train_set = windowed_dataset(x_train, window_size=60, batch_size=250, shuffle_buffer_size=shuffle_buffer_size)

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv1D(filters=60, kernel_size=5, strides=1, padding="causal",activation="relu", input_shape=[None, 1]),

  tf.keras.layers.LSTM(60, return_sequences=True),

  tf.keras.layers.BatchNormalization(),  

  tf.keras.layers.LSTM(60),

  tf.keras.layers.BatchNormalization(),  

  tf.keras.layers.Dense(30, activation="relu"),

  tf.keras.layers.BatchNormalization(),  

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

rnn_forecast = rnn_forecast[split_time - window_size:-1, -1]
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, rnn_forecast)
tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

loss=history.history['loss']



epochs=range(len(loss)) # Get number of epochs



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r')

plt.title('Training loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(["Loss"])



plt.figure()



zoomed_loss = loss[200:]

zoomed_epochs = range(200,500)





#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(zoomed_epochs, zoomed_loss, 'r')

plt.title('Training loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(["Loss"])



plt.figure()