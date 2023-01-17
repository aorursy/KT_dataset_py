import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import tensorflow as tf

from tensorflow import keras
train = pd.read_csv('/kaggle/input/grocery-sales-forecast-weekend-hackathon/Grocery_Sales_ParticipantsData/Train.csv')

test = pd.read_csv('/kaggle/input/grocery-sales-forecast-weekend-hackathon/Grocery_Sales_ParticipantsData/Test.csv')
train.head()
train.shape, test.shape
series = train.GrocerySales.values

time = np.arange(692, dtype="float32")
def plot_series(time, series, format="-", start=0, end=None):

    plt.plot(time[start:end], series[start:end], format)

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.grid(True)
plt.figure(figsize=(10, 6))

plot_series(time, series)

plt.show()
split_time = 600

time_train = time[:split_time]

x_train = series[:split_time]

time_valid = time[split_time:]

x_valid = series[split_time:]

plt.figure(figsize=(10, 6))

plot_series(time_train, x_train)

plt.show()



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plt.show()
naive_forecast = series[split_time - 1:-1]

plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, naive_forecast)
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid, start=0, end=150)

plot_series(time_valid, naive_forecast, start=1, end=151)
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())

print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
def moving_average_forecast(series, window_size):

    forecast = []

    for time in range(len(series) - window_size):

        forecast.append(series[time:time + window_size].mean())

    return np.array(forecast)



moving_avg = moving_average_forecast(series, 30)[split_time - 30:]



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, moving_avg)



print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())

print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
diff_series = (series[365:] - series[:-365])

diff_time = time[365:]



plt.figure(figsize=(10, 6))

plot_series(diff_time, diff_series)

plt.show()
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]



plt.figure(figsize=(10, 6))

plot_series(time_valid, diff_series[split_time - 365:])

plot_series(time_valid, diff_moving_avg)

plt.show()
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, diff_moving_avg_plus_past)

plt.show()
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, diff_moving_avg_plus_smooth_past)

plt.show()
print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())

print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
window_size = 20

batch_size = 32

shuffle_buffer_size = 1000
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    dataset = tf.data.Dataset.from_tensor_slices(series)

    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))

    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

print(dataset)

l0 = tf.keras.layers.Dense(1, input_shape=[window_size])

model = tf.keras.models.Sequential([l0])



model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model.fit(dataset,epochs=100,verbose=0)



print("Layer weights {}".format(l0.get_weights()))
forecast = []



for time in range(len(series) - window_size):

    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))



forecast = forecast[split_time-window_size:]

results = np.array(forecast)[:, 0, 0]





plt.figure(figsize=(10, 6))



plot_series(time_valid, x_valid)

plot_series(time_valid, results)
print(np.sqrt(keras.metrics.mean_squared_error(x_valid, results).numpy()))

print(keras.metrics.mean_absolute_error(x_valid, results).numpy())
tf.keras.backend.clear_session()

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)



model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 

    tf.keras.layers.Dense(10, activation="relu"), 

    tf.keras.layers.Dense(1)

])



model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

history = model.fit(dataset,epochs=100,verbose=0)
loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training Loss')

plt.show()
loss = history.history['loss']

epochs = range(90, len(loss))

plot_loss = loss[90:]

print(plot_loss)

plt.plot(epochs, plot_loss, 'b', label='Training Loss')

plt.show()
forecast = []

for time in range(len(series) - window_size):

    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))



forecast = forecast[split_time-window_size:]

results = np.array(forecast)[:, 0, 0]





plt.figure(figsize=(10, 6))



plot_series(time_valid, x_valid)

plot_series(time_valid, results)
print(np.sqrt(keras.metrics.mean_squared_error(x_valid, results).numpy()))

print(keras.metrics.mean_absolute_error(x_valid, results).numpy())