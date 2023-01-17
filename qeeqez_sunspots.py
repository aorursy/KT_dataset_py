# Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



input_file='/kaggle/input/sunspots/Sunspots.csv'
# Function to plot series

def plot_series(time, series, format = "-", start = 0, end = None):

    plt.plot(time[start:end], series[start:end], format)

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.grid(True)
# Load the dataset

# Convert the dataset into time series numpy array

import csv

time_step = []

temps = []



df = pd.read_csv(input_file)

df = df.drop(columns=['Unnamed: 0'])

df.Date = pd.to_datetime(df.Date)

df = df.set_index("Date")

df['Date'] = df.index

df = df.rename(columns={'Monthly Mean Total Sunspot Number': 'Sunspot'})

df.head()
sunspots = df['Sunspot']

dates = df['Date']
with plt.style.context('ggplot'):

    plt.figure(figsize=(16, 8))

    plt.plot(sunspots)

    plot_series(dates, sunspots)
from pylab import rcParams

from statsmodels.tsa.seasonal import seasonal_decompose 



rcParams.update({

    'figure.figsize': (16, 8),

    'font.size': 16,

})



with plt.style.context('ggplot'):

  a = seasonal_decompose(sunspots, model = "additive")

  a.plot()
from pandas.plotting import autocorrelation_plot

with plt.style.context('ggplot'):

  plt.figure(figsize=(16, 8))

  autocorrelation_plot(df['Sunspot'])

  plt.show()
# Define parameters

split_time = int(len(df)*0.85)

time_train = dates[:split_time]

X_train = sunspots[:split_time]

time_valid = dates[split_time:]

X_valid = sunspots[split_time:]



window_size = 30

batch_size = 32

shuffle_buffer_size = 1000



print(f"Train: {len(X_train)}; Test: {len(X_valid)}")
# Prepare the training set

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
from tqdm import tqdm

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA



def calc_arima(X_train, X_test, order):

    history = [x for x in X_train]

    predictions = list()

    for t in tqdm(range(len(X_test))):

        model = ARIMA(history, order=order)

        result = model.fit(disp=0)

        prediction = result.forecast()[0]

        predictions.append(prediction)

        history.append(X_test[t])

    error = mean_squared_error(X_test, predictions)

    return X_test, predictions, error
temp_test, predictions, arima_error = calc_arima(X_train, X_valid, (5,1,0))
print(f"ARIMA MSE: {arima_error}")
predictions = pd.Series(predictions, index=time_valid)

valid = pd.Series(X_valid, index=time_valid)
with plt.style.context('ggplot'):

    plt.figure(figsize=(16, 8))

    plt.plot(valid, label="Real")

    plt.plot(predictions, label="Predicted")

    labels = ['Real', 'Predicted']

    plt.legend(labels)

    plt.show()
# Define the model



tf.keras.backend.clear_session()

tf.random.set_seed(0)

np.random.seed(0)



train_set = windowed_dataset(X_train, window_size=60, batch_size=64, shuffle_buffer=shuffle_buffer_size)



def create_model():

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(32, 5, strides=1, padding="same", activation="relu", input_shape=[None, 1]),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

        tf.keras.layers.Dense(32, activation="relu"),

        tf.keras.layers.Dense(16, activation="relu"),

        tf.keras.layers.Dense(1),

    ])

    model.compile(

        loss=tf.keras.losses.Huber(),

        optimizer='adam',

        metrics=["mae"]

    )

    return model



model = create_model()

model.summary()
history = model.fit(train_set,epochs=100)
# Predict the result



series = np.array(df['Sunspot'])

rnn_forecast = model_forecast(model,  series[..., np.newaxis], window_size)

rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
# Evaluate the model using MAE

lstm_error = mean_squared_error(X_valid, rnn_forecast)

print(f"LSTM MSE: {lstm_error}")
# Visualising the result



with plt.style.context('ggplot'):

    plt.figure(figsize=(16, 8))

    plot_series(time_valid, X_valid)

    plot_series(time_valid, rnn_forecast)

    labels = ['Real', 'Predicted']

    plt.legend(labels)

    plt.show()