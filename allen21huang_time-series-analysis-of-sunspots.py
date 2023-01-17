# everything can be pip install

import csv

import datetime

import warnings

import numpy as np

import pandas as pd

import statsmodels.api as sm   

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras

from itertools import product    

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from matplotlib import dates as mpl_dates
warnings.filterwarnings(action = 'ignore')
# define a function to plot time series data

def plot_series(time, series, col = 'dodgerblue', lab = 'original', format="-", start=0, end=None):

    plt.style.use('seaborn')

    plt.plot(time[start:end], series[start:end], format, color = col, label = lab)

    plt.xlabel("Time")

    plt.ylabel("Series")

    # display the grid

    plt.grid(True)

    # got current figure, then autoformat date

    plt.gcf().autofmt_xdate() 

    # format datetime

    date_formate = mpl_dates.DateFormatter('%b/%d/%Y') 

    # set the format to out x-axis, gca is the get current axis

    plt.gca().xaxis.set_major_formatter(date_formate)

    plt.tight_layout() 

    plt.legend(loc = 'best')
df = pd.read_csv('/kaggle/input/Sunspots.csv')

del df['Unnamed: 0']

df.head()

# check for missing value using df.isnull(), there isn't any NaN value in this dataframe
timeseries = df

timeseries['Date'] = pd.to_datetime(df['Date'])

timeseries = df.set_index(df['Date'])

del timeseries['Date']

timeseries.head()
time_step = []

sunspots = []

for time, value in zip(df['Date'],df['Monthly Mean Total Sunspot Number']):

    time_step.append(time)

    sunspots.append(float(value))



# plot our data

plt.figure(figsize=(10, 6))



plot_series(time_step, sunspots)
# zoom into the plot, the seasonality is roughly 11 years

plt.figure(figsize=(10, 6))

plot_series(time_step, sunspots, start=0, end=300)
# split data into validation and training datasets

split_time = int(len(time_step)*0.8)

time_train = time_step[:split_time]

x_train = sunspots[:split_time]

time_valid = time_step[split_time:]

x_valid = sunspots[split_time:]
# becasue data is from time_step and sunspots, here our data are in list type

type(x_train)
# I will also split it this way, and now train and vaild dataset are daraframe

split_time = int(len(time_step)*0.8)

train = timeseries[:split_time]

valid = timeseries[split_time:]
train.head()
train.index
naive_forecast = sunspots[split_time - 1:-1]



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, naive_forecast, col = 'coral', lab = 'naive forecast')
# zoom into to figure to avoid overlap

plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid, start=0, end=150)

plot_series(time_valid, naive_forecast, start=1, end=151, col = 'coral', lab = 'naive forecast')

# the orange data is just one step after the blue data
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
# Firstly, define a function for moving average forecast

# window size means forecasts the means for the last few values

def moving_average_forecast(series, window_size):

    forecast = []

    for time in range(len(series) - window_size):

        forecast.append(series[time:time + window_size].mean())

    return np.array(forecast)
# turn data into numpy array 

series = np.array(sunspots)

time = np.array(time_step)

# here, the windex size is set to be 30

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, moving_avg, col = 'coral', lab = 'moving average forecast')
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
# difference between time t and 132 timesteps before that 

# after finding to optimal parameter, seasonality is set to be 507

seasonality = 507

diff_series = (series[seasonality:] - series[:-seasonality])

diff_time = time[seasonality:]



plt.figure(figsize=(10, 6))

plot_series(diff_time, diff_series)

plt.show()
# try window size 20

# after finding to optimal parameter, window size is set to be 5

diff_moving_avg = moving_average_forecast(diff_series, 5)[split_time - seasonality - 5:]



plt.figure(figsize=(10, 6))

plot_series(time_valid, diff_series[split_time - seasonality:])

plot_series(time_valid, diff_moving_avg, col = 'coral')

plt.show()
# adding back the past value

diff_moving_avg_plus_past = series[split_time - seasonality:-seasonality] + diff_moving_avg



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, diff_moving_avg_plus_past, col = 'coral')

plt.show()
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - seasonality-5:-seasonality+5], 10) + diff_moving_avg



plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)

plot_series(time_valid, diff_moving_avg_plus_smooth_past, col = 'coral', lab = 'diff_moving_avg_plus_smooth_past')

plt.show()
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
seasonpara = []

mae = []

window = []

for window_size in range(1,10):

    for seasonality in range(500, 600):

        diff_series = (series[seasonality:] - series[:-seasonality])

        diff_time = time[seasonality:]

        diff_moving_avg = moving_average_forecast(diff_series, window_size)[split_time - seasonality - window_size:]

        diff_moving_avg_plus_past = series[split_time - seasonality:-seasonality] + diff_moving_avg

        diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - seasonality-5:-seasonality+5], 10) + diff_moving_avg

        seasonpara.append(seasonality)

        mae.append(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())

        window.append(window_size)
for i in mae:

    if i == min(mae):

        print(f'The optimal seaonality is: {seasonpara[mae.index(i)]}')

        print(f'The optimal window size is: {window[mae.index(i)]}')

        print(f'The optimal MAE is: {min(mae)}')
timeseries.head()
def plot_rol(timeseriesdata, size):

    plt.figure(figsize=(15, 7))

    plt.style.use('seaborn')

    # rolling statistics

    rol_mean = timeseriesdata.rolling(window=size).mean()

    rol_std = timeseriesdata.rolling(window=size).std()

    plt.plot(timeseriesdata, color='dodgerblue', label='Original')

    plt.plot(rol_mean, color='red', label='Rolling Mean')

    plt.plot(rol_std, color='green', label='Rolling standard deviation')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    # let the axis off for convenience

    plt.axis('off')

    plt.show()
plot_rol(timeseries, 132)
plot_rol(timeseries.diff(1), 132)
# define a function for adfuller test

def teststationarity(ts):

    print('result of dickey-fuller test:')

    dftest = adfuller(ts['Monthly Mean Total Sunspot Number'], autolag = 'AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    return dfoutput
teststationarity(timeseries)
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(timeseries.dropna(), freq = 132)



# get the trend, seasonality and noise 

trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.figure(figsize=(12,8))

plt.subplot(411)

plt.plot(timeseries, label='Original', color="dodgerblue")

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label='Trend', color="green")

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality', color="red")

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label='Residuals', color="orange")

plt.legend(loc='best')

plt.tight_layout()
fig, ax = plt.subplots(figsize=(16,3))

plot_acf(timeseries,ax=ax, lags = 200, color="dodgerblue");



fig, ax = plt.subplots(figsize=(16,3))

plot_pacf(timeseries,ax=ax, lags = 200, color="dodgerblue")

plt.show()
autocorrelation_plot(timeseries, color='dodgerblue')
# the first value is NaN when use a diff

autocorrelation_plot(timeseries.diff(1)[1:], color='dodgerblue')

# in this case, the model is overfitting 
atrain = train.resample('A').sum()

avalid = valid.resample('A').sum()

avalid.head()
# Initial approximation of parameters

Qs = range(0, 3)

qs = range(0, 3)

Ps = range(0, 3)

ps = range(0, 3)

D=1

d=1

parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

len(parameters_list)



# Model Selection

results = []

best_aic = float("inf")

warnings.filterwarnings('ignore')

for param in parameters_list:

    try:

        model=sm.tsa.statespace.SARIMAX(atrain['Monthly Mean Total Sunspot Number'], order=(param[0], d, param[1]), 

                                        seasonal_order=(param[2], D, param[3], 11)).fit(disp=-1)

    except ValueError:

        print('wrong parameters:', param)

        continue

    aic = model.aic

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = param

    results.append([param, model.aic])
best_model.summary()
best_model.plot_diagnostics(figsize=(15, 12))

plt.show()
plt.plot(atrain,color = 'dodgerblue',label = 'trian')

plt.plot(avalid, color = 'r',label = 'valid')

best_model.forecast(len(avalid)).plot(color = 'orange',label = 'forecast')

plt.legend(loc = 'best')

plt.show()
# define some hyper-parameter

window_size = 60

batch_size = 32

shuffle_buffer_size = 1000



# turn a series into a dataset which we can train on

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    # create a dataset from series

    dataset = tf.data.Dataset.from_tensor_slices(series)

    # slice the data up into appropriate windows

    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # flattened into chunks in the size of our window size + 1

    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))

    # batched into the selected batch size and returned

    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

# three layer of 20,10 and 1 neurons, input shape is the size of window 

model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(20, input_shape=[window_size], activation="relu"), 

    tf.keras.layers.Dense(10, activation="relu"),

    tf.keras.layers.Dense(1)

])



model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9))

# ignore the epoch by epoch output by setting verbose = 0

history = model.fit(dataset,epochs=200,verbose=0)

model.summary()
loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training Loss')

plt.show()
series = np.array(sunspots)

time = np.array(time_step)

forecast=[]



for time in range(len(series) - window_size):

    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))



forecast = forecast[split_time-window_size:]

results = np.array(forecast)[:, 0, 0]



plt.figure(figsize=(10, 6))



plot_series(time_valid, x_valid)

plot_series(time_valid, results, col = 'coral')
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
def model_forecast(model, series, window_size):

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size, shift=1, drop_remainder=True)

    ds = ds.flat_map(lambda w: w.batch(window_size))

    ds = ds.batch(32).prefetch(1)

    forecast = model.predict(ds)

    return forecast



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



model = tf.keras.models.Sequential([

  tf.keras.layers.Conv1D(filters=32, kernel_size=5,

                      strides=1, padding="causal",

                      activation="relu",

                      input_shape=[None, 1]),

  # Sequence to Sequence for LSTM 

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.Dense(30, activation="relu"),

  tf.keras.layers.Dense(10, activation="relu"),

  tf.keras.layers.Dense(1),

  tf.keras.layers.Lambda(lambda x: x * 400)

])



# twick the learning rate, use the optimal learning rate instead of one 

lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])

history = model.fit(train_set, epochs=100, callbacks=[lr_schedule],verbose = 0)
plt.semilogx(history.history["lr"], history.history["loss"])

plt.axis([1e-8, 1e-4, 0, 60])

# y is the loss and x is the learning rate 

# set the lr to be the optimal, where the loss is this minimum
# this clears any internal variables, which makes it easy for us to experiment without models impacting later versions of themselves.

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

history = model.fit(train_set,epochs=500,verbose=0)
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