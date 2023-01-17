import numpy as np

import pandas as pd

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None):

    plt.plot(time[start:end], series[start:end], format)

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.grid(True)

print(tf.__version__)

# make sure that tensorflow version is 2.0 or above
# using pandas to import data instead of CSV



data = pd.read_csv('/kaggle/input/sunspots/Sunspots.csv')

data.head()
series = data['Monthly Mean Total Sunspot Number'].to_numpy()

series[:5]
time = data['Unnamed: 0'].to_numpy()

time[:5]
plt.figure(figsize=(10, 6))

plot_series(time, series)
split_time = 2500 #changed from 3000 to 2500 to get more validation data

time_train = time[:split_time]

x_train = series[:split_time]

time_valid = time[split_time:]

x_valid = series[split_time:]



window_size = 64 

batch_size = 256

shuffle_buffer_size = 1000
# convert series to an input set for RNN/LSTM

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

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



model.summary()
# running for only 100 epochs to find the optimal learning rate



lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
plt.semilogx(history.history["lr"], history.history["loss"])

plt.axis([1e-8, 1e-4, 0, 60])
# update learning rate to 1e-5

# update epoch to 500

# updated LSTM, Dense layers



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





optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

history = model.fit(train_set,epochs=500)
def model_forecast(model, series, window_size):

    # preparing ds

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size, shift=1, drop_remainder=True)

    ds = ds.flat_map(lambda w: w.batch(window_size))

    ds = ds.batch(32).prefetch(1)

    # forecast

    forecast = model.predict(ds)

    return forecast



rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)

rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
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
print(rnn_forecast)