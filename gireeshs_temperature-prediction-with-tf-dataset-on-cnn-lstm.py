import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 20)



import tensorflow as tf



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv")

data.head()
data['City'].value_counts()
chennai = data[data["City"] == "Chennai (Madras)"]

chennai.head()
chennai["Year"].value_counts()
"""-99 is put in place of missing values. 

We will have to forward fill with the last non missing value before -99

"""

chennai["AvgTemperature"] = np.where(chennai["AvgTemperature"] == -99, np.nan, chennai["AvgTemperature"])

chennai.isnull().sum()
chennai["AvgTemperature"] = chennai["AvgTemperature"].ffill()

chennai.isnull().sum()
chennai.dtypes

chennai["Time_steps"] = pd.to_datetime((chennai.Year*10000 + chennai.Month*100 + chennai.Day).apply(str),format='%Y%m%d')

chennai.head()
def plot_series(time, series, format="-", start=0, end=None):

    """to plot the series"""

    plt.plot(time[start:end], series[start:end], format)

    plt.xlabel("Year")

    plt.ylabel("Temprature")

    plt.grid(True)
time_step = chennai["Time_steps"].tolist()

temprature = chennai["AvgTemperature"].tolist()



series = np.array(temprature)

time = np.array(time_step)

plt.figure(figsize=(10, 6))

plot_series(time, series)
plt.figure(figsize=(10, 6))

plot_series(time[-365:], series[-365:])
split_time = 8000

time_train = time[:split_time]

x_train = series[:split_time]

time_valid = time[split_time:]

x_valid = series[split_time:]
naive_forecast = series[split_time - 1:-1]
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, naive_forecast)
#Zoom in and see only few points

plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid, start=0, end=150)

plot_series(time_valid, naive_forecast, start=1, end=151)
print(tf.keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())

print(tf.keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
def moving_average_forecast(series, window_size):

    """Forecasts the mean of the last few values.

     If window_size=1, then this is equivalent to naive forecast"""

    forecast = []

    for time in range(len(series) - window_size):

        forecast.append(series[time:time + window_size].mean())

    return np.array(forecast)
moving_avg = moving_average_forecast(series, 30)[split_time - 30:]



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, moving_avg)
print(tf.keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())

print(tf.keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
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
print(tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())

print(tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, diff_moving_avg_plus_smooth_past)

plt.show()
print(tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())

print(tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
series1 = tf.expand_dims(series, axis=-1)

ds = tf.data.Dataset.from_tensor_slices(series1[:20])

for val in ds:

    print(val.numpy())



dataset = ds.window(5, shift=1)

for window_dataset in dataset:

    for val in window_dataset:

        print(val.numpy(), end=" ")

    print()
dataset = ds.window(5, shift=1, drop_remainder=True)

for window_dataset in dataset:

    for val in window_dataset:

        print(val.numpy(), end=" ")

    print()
dataset = ds.window(5, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(5))

for window in dataset:

    print(window.numpy())
dataset = ds.window(5, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(5))

dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

for x,y in dataset:

    print(x.numpy(), y.numpy())
dataset = ds.window(5, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(5))

dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

dataset = dataset.shuffle(buffer_size=10)

for x,y in dataset:

    print(x.numpy(), y.numpy())
dataset = ds.window(5, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(5))

dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

dataset = dataset.shuffle(buffer_size=10)

dataset = dataset.batch(2).prefetch(1)

for x,y in dataset:

    print("x = ", x.numpy())

    print("y = ", y.numpy())

    print("*"*25)
window_size = 60

batch_size = 32

shuffle_buffer_size = 1000
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    """

    To create a window dataset given a numpy as input

    

    Returns: A prefetched tensorflow dataset

    """

    series = tf.expand_dims(series, axis=-1)

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)

    ds = ds.flat_map(lambda w: w.batch(window_size + 1))

    ds = ds.shuffle(shuffle_buffer)

    ds = ds.map(lambda w: (w[:-1], w[1:]))

    return ds.batch(batch_size).prefetch(1)
tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)

window_size = 64

batch_size = 256

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

print(train_set)

print(x_train.shape)



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



lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))



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

train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)

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





optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

history = model.fit(train_set,epochs=500)
def model_forecast(model, series, window_size):

    """

    Given a model object and a series for it to predict, this function will return the prediction

    """

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size, shift=1, drop_remainder=True)

    ds = ds.flat_map(lambda w: w.batch(window_size))

    ds = ds.batch(32).prefetch(1)

    forecast = model.predict(ds)

    return forecast
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)

rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, rnn_forecast)
tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
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