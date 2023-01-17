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
calender=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train_evaluation=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

sample_submission=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

sell_prices=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sales_train_validation=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
sales_train_validation.head()
# sell_prices=sell_prices[:100]

len(sell_prices)

sell_prices.head()
# sales_train_evaluation=sales_train_evaluation[:100]

print(len(sales_train_evaluation))

sales_train_evaluation.head()


print(len(calender))

calender.head()
train=sell_prices[:100].merge(calender, on='wm_yr_wk')
train.head()
train_time=train['month']
train_value=train['store_id']
train_value.isnull().sum()
store_sum = train.groupby(['store_id'],).sum().T.reset_index(drop = True)

# from statsmodels.tsa.seasonal import seasonal_decompose

days_per_week = train['d']



time_series = train['wm_yr_wk']

# sj_sc = seasonal_decompose(time_series, period = days_per_week)

# sj_sc.plot()



# plt.show()
def plot_series(time, series, format="-", start=0, end=None, label=None):

    plt.plot(time[start:end], series[start:end], format, label=label)

    plt.xlabel("Time")

    plt.ylabel("Value")

    if label:

        plt.legend(fontsize=14)

    plt.grid(True)
import numpy as np

import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))



plot_series(days_per_week, time_series)

plt.show()
time = days_per_week

series = time_series

split_time = 500

time_train = time[:split_time]

x_train = series[:split_time]

time_valid = time[split_time:]

x_valid = series[split_time:]
naive_forecast = series[split_time - 1:-1]
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid, label="Series")

plot_series(time_valid, naive_forecast, label="Forecast")
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid, start=0, end=150, label="Series")

plot_series(time_valid, naive_forecast, start=1, end=151, label="Forecast")
errors = naive_forecast - x_valid

abs_errors = np.abs(errors)

mae = abs_errors.mean()

mae
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



keras = tf.keras
def window_dataset(series, window_size, batch_size=32,

                   shuffle_buffer=1000):

    dataset = tf.data.Dataset.from_tensor_slices(series)

    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset
def model_forecast(model, series, window_size):

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size, shift=1, drop_remainder=True)

    ds = ds.flat_map(lambda w: w.batch(window_size))

    ds = ds.batch(32).prefetch(1)

    forecast = model.predict(ds)

    return forecast
keras.backend.clear_session()

tf.random.set_seed(42)

np.random.seed(42)



window_size = 30

train_set = window_dataset(x_train, window_size)



model = keras.models.Sequential([

  keras.layers.Dense(10, activation="relu", input_shape=[window_size]),

  keras.layers.Dense(10, activation="relu"),

  keras.layers.Dense(1)

])



lr_schedule = keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-7 * 10**(epoch / 20))

optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)

model.compile(loss=keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
plt.semilogx(history.history["lr"], history.history["loss"])

plt.axis([e-5, 5e-3, 0, 30])
keras.backend.clear_session()

tf.random.set_seed(42)

np.random.seed(42)



window_size = 30

train_set = window_dataset(x_train, window_size)

valid_set = window_dataset(x_valid, window_size)



model = keras.models.Sequential([

  keras.layers.Dense(10, activation="relu", input_shape=[window_size]),

  keras.layers.Dense(10, activation="relu"),

  keras.layers.Dense(1)

])



optimizer = keras.optimizers.SGD(lr=1e-6, momentum=0.9)

model.compile(loss=keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

early_stopping = keras.callbacks.EarlyStopping(patience=10)

model.fit(train_set, epochs=500,

          validation_data=valid_set,

          callbacks=[early_stopping])
dense_forecast = model_forecast(

    model,

    series[split_time - window_size:-1],

    window_size)[:, 0]
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, dense_forecast)
keras.metrics.mean_absolute_error(x_valid, dense_forecast).numpy()
keras.backend.clear_session()

tf.random.set_seed(42)

np.random.seed(42)



window_size = 30

train_set = window_dataset(x_train, window_size, batch_size=128)



model = keras.models.Sequential([

  keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),

                      input_shape=[None]),

  keras.layers.SimpleRNN(100, return_sequences=True),

  keras.layers.SimpleRNN(100),

  keras.layers.Dense(1),

  keras.layers.Lambda(lambda x: x * 200.0)

])

lr_schedule = keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-7 * 10**(epoch / 20))

optimizer = keras.optimizers.SGD(lr=1e-7, momentum=0.9)

model.compile(loss=keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
plt.semilogx(history.history["lr"], history.history["loss"])

plt.axis([e-7, 1e-4, 0, 30])
keras.backend.clear_session()

tf.random.set_seed(42)

np.random.seed(42)



window_size = 30

train_set = window_dataset(x_train, window_size, batch_size=128)

valid_set = window_dataset(x_valid, window_size, batch_size=128)



model = keras.models.Sequential([

  keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),

                      input_shape=[None]),

  keras.layers.SimpleRNN(100, return_sequences=True),

  keras.layers.SimpleRNN(100),

  keras.layers.Dense(1),

  keras.layers.Lambda(lambda x: x * 200.0)

])

optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)

model.compile(loss=keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

early_stopping = keras.callbacks.EarlyStopping(patience=50)

model_checkpoint = keras.callbacks.ModelCheckpoint(

    "my_checkpoint", save_best_only=True)

model.fit(train_set, epochs=500,

          validation_data=valid_set,

          callbacks=[early_stopping, model_checkpoint])
model = keras.models.load_model("my_checkpoint")
rnn_forecast = model_forecast(

    model,

    series[split_time - window_size:-1],

    window_size)[:, 0]
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, rnn_forecast)
keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()


def sequential_window_dataset(series, window_size):

    series = tf.expand_dims(series, axis=-1)

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)

    ds = ds.flat_map(lambda window: window.batch(window_size + 1))

    ds = ds.map(lambda window: (window[:-1], window[1:]))

    return ds.batch(1).prefetch(1)



class ResetStatesCallback(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs):

        self.model.reset_states()
keras.backend.clear_session()

tf.random.set_seed(42)

np.random.seed(42)



window_size = 30

train_set = sequential_window_dataset(x_train, window_size)



model = keras.models.Sequential([

  keras.layers.LSTM(100, return_sequences=True, stateful=True,

                    batch_input_shape=[1, None, 1]),

  keras.layers.LSTM(100, return_sequences=True, stateful=True),

  keras.layers.Dense(1),

  keras.layers.Lambda(lambda x: x * 200.0)

])

lr_schedule = keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8 * 10**(epoch / 20))

reset_states = ResetStatesCallback()

optimizer = keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss=keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

history = model.fit(train_set, epochs=100,

                    callbacks=[lr_schedule, reset_states])
plt.semilogx(history.history["lr"], history.history["loss"])

plt.axis([e-8, 1e-4, 0, 30])
keras.backend.clear_session()

tf.random.set_seed(42)

np.random.seed(42)



window_size = 30

train_set = sequential_window_dataset(x_train, window_size)

valid_set = sequential_window_dataset(x_valid, window_size)



model = keras.models.Sequential([

  keras.layers.LSTM(100, return_sequences=True, stateful=True,

                         batch_input_shape=[1, None, 1]),

  keras.layers.LSTM(100, return_sequences=True, stateful=True),

  keras.layers.Dense(1),

  keras.layers.Lambda(lambda x: x * 200.0)

])

optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)

model.compile(loss=keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

reset_states = ResetStatesCallback()

model_checkpoint = keras.callbacks.ModelCheckpoint(

    "my_checkpoint.h5", save_best_only=True)

early_stopping = keras.callbacks.EarlyStopping(patience=50)

model.fit(train_set, epochs=500,

          validation_data=valid_set,

          callbacks=[early_stopping, model_checkpoint, reset_states])
model = keras.models.load_model("my_checkpoint.h5")
rnn_forecast = model.predict(series[np.newaxis, :, np.newaxis])

rnn_forecast = rnn_forecast[0, split_time - 1:-1, 0]
plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, rnn_forecast)
keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
def model_forecast(model, series, window_size):

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size, shift=1, drop_remainder=True)

    ds = ds.flat_map(lambda w: w.batch(window_size))

    ds = ds.batch(32).prefetch(1)

    forecast = model.predict(ds)

    return forecast
keras.backend.clear_session()

tf.random.set_seed(42)

np.random.seed(42)



window_size = 64

train_set = seq2seq_window_dataset(x_train, window_size,

                                   batch_size=128)

valid_set = seq2seq_window_dataset(x_valid, window_size,

                                   batch_size=128)



model = keras.models.Sequential()

model.add(keras.layers.InputLayer(input_shape=[None, 1]))

for dilation_rate in (1, 2, 4, 8, 16, 32):

    model.add(

      keras.layers.Conv1D(filters=32,

                          kernel_size=2,

                          strides=1,

                          dilation_rate=dilation_rate,

                          padding="causal",

                          activation="relu")

    )

model.add(keras.layers.Conv1D(filters=1, kernel_size=1))

optimizer = keras.optimizers.Adam(lr=3e-4)

model.compile(loss=keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])



model_checkpoint = keras.callbacks.ModelCheckpoint(

    "my_checkpoint.h5", save_best_only=True)

early_stopping = keras.callbacks.EarlyStopping(patience=50)

history = model.fit(train_set, epochs=500,

                    validation_data=valid_set,

                    callbacks=[early_stopping, model_checkpoint])
model = keras.models.load_model("my_checkpoint.h5")