# The Imports

import os

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
Data = pd.read_csv("../input/sunspots/Sunspots.csv")
Data.head()
Data.isna().sum()
Sunspots = Data['Monthly Mean Total Sunspot Number']

series = np.array(Sunspots)

time = np.arange(0, 3235)
# The function to plot series data

def plotter(time, series, format="-", start=0, end=None):

    plt.plot(time[start:end], series[start:end], format)

    plt.xlabel("Time (Months)")

    plt.ylabel("Sunspots")

    plt.grid(True)
# Plotting the data to have a 'first look'

plt.figure(figsize = (20, 6))

plotter(time, series)
# Defining the parameters

window_size = 100

batch_size = 256

shuffle_buffer_size = 1000
# Windowing the dataset

def train_data_pipeline(series, window_size, batch_size, shuffle_buffer_size):

    series = tf.expand_dims(series, axis=-1)

    data = tf.data.Dataset.from_tensor_slices(series)

    data = data.window(window_size+1, shift=1, drop_remainder=True)

    data = data.flat_map(lambda w: w.batch(batch_size))

    data = data.shuffle(shuffle_buffer_size)

    data = data.map(lambda w: (w[:-1], w[1:]))

    return data.batch(batch_size).prefetch(1)

    

train = train_data_pipeline(series, window_size, batch_size, shuffle_buffer_size)
# Something new that I designed out of the blue.

tf.keras.backend.clear_session()



# The model

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv1D(filters=100, kernel_size=3, padding='causal', activation='relu', input_shape=[None, 1]),

    tf.keras.layers.Reshape((100, 100, 1)),

    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='valid', activation='relu'),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu'),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(8),

    tf.keras.layers.Dense(1),

    tf.keras.layers.Lambda(lambda x: x*400)

])



model.summary()
# Compiling nad Training the model

model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=['mae'])

history = model.fit(train, epochs=100, verbose=0)
# Plotting loss values

plt.plot(history.history['loss'][50:])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()



# Plotting mae values

plt.plot(history.history['mae'][50:])

plt.title('Model Mae')

plt.ylabel('Mae')

plt.xlabel('Epoch')

plt.show()
# That thing didn't work. So, here's a different model.

tf.keras.backend.clear_session()



# The model

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='causal', activation='relu', input_shape=[None, 1]),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.Dense(32, activation="relu"),

  tf.keras.layers.Dense(16, activation="relu"),

  tf.keras.layers.Dense(1),

  tf.keras.layers.Lambda(lambda x: x * 400)

])



model.summary()
# Compiling and Training the Model

model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=["mae"])

history = model.fit(train, epochs=500, verbose=0)
# Plotting loss values

plt.plot(history.history['loss'][50:])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()



# Plotting mae values

plt.plot(history.history['mae'][50:])

plt.title('Model Mae')

plt.ylabel('Mae')

plt.xlabel('Epoch')

plt.show()
# Okay. That looks good.



# Windowing the data for testing

def test_data_pipeline(series):

    series = tf.expand_dims(series, axis=-1)

    data = tf.data.Dataset.from_tensor_slices(series)

    data = data.window(window_size, shift=1, drop_remainder=True)

    data = data.flat_map(lambda w: w.batch(window_size))

    data = data.batch(batch_size).prefetch(1)

    return data
# Predicting on the same dataset

test = test_data_pipeline(series)

forecast = model.predict(test)

forecast = forecast[:, -1, 0]
# Forecasting sunspots using the trained model

time_valid = time[window_size+1:]

series_ori = series[window_size+1:]

forecast = np.reshape(forecast, (-1))

forecast = forecast[:-1]

plt.figure(figsize=(20, 6))

plotter(time_valid[2000:2500], series_ori[2000:2500])

plotter(time_valid[2000:2500], forecast[2000:2500])