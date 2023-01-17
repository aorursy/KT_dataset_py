# Import Libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

import tensorflow as tf

from tensorflow import keras



# plt.style.available

plt.style.use("seaborn-whitegrid")
df = pd.read_csv('../input/sandp500/all_stocks_5yr.csv')

df.head()
df.describe()
aal = df.loc[df['Name'] == 'AAL']

aal.head()
aal.info() 
# Create a copy to avoid the SettingWarning .loc issue 

aal_df = aal.copy()

# Change to datetime datatype.

aal_df.loc[:, 'date'] = pd.to_datetime(aal.loc[:,'date'], format="%Y/%m/%d")
aal_df.info()
# Simple plotting of Amazon Stock Price

# First Subplot

f, (ax1) = plt.subplots(1, figsize=(20,5))

ax1.plot(aal_df["date"], aal_df["close"])

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Stock Price")

ax1.set_title("AAL Close Price History")
def univariate_data(dataset, start_index, end_index, history_size, target_size):

  data = []

  labels = []



  start_index = start_index + history_size

  if end_index is None:

    end_index = len(dataset) - target_size



  for i in range(start_index, end_index):

    indices = range(i-history_size, i)

    # Reshape data from (history_size,) to (history_size, 1)

    data.append(np.reshape(dataset[indices], (history_size, 1)))

    labels.append(dataset[i+target_size])

  return np.array(data), np.array(labels)



train_split = 900
aal_data = aal_df['close']

aal_data.index = aal_df['date']

aal_data.head()
aal_data.plot(subplots=True)
aal_data = aal_data.values
aal_train_mean = aal_data[:train_split].mean()

aal_train_std = aal_data[:train_split].std()



aal_data = (aal_data-aal_train_mean)/aal_train_std
data_past_history = 64

data_future_target = 0



x_train, y_train = univariate_data(aal_data, 0, train_split,

                                        data_past_history,

                                        data_future_target)

x_val, y_val = univariate_data(aal_data, train_split, None,

                                       data_past_history,

                                       data_future_target)
print ('Single window of past history')

print (x_train[0])

print ('\n Target temperature to predict')

print (y_train[0])
def create_time_steps(length):

  return list(range(-length, 0))
def show_plot(plot_data, delta, title):

  labels = ['History', 'True Future', 'Model Prediction']

  marker = ['.-', 'rx', 'go']

  time_steps = create_time_steps(plot_data[0].shape[0])

  if delta:

    future = delta

  else:

    future = 0



  plt.title(title)

  for i, x in enumerate(plot_data):

    if i:

      plt.plot(future, plot_data[i], marker[i], markersize=10,

               label=labels[i])

    else:

      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])

  plt.legend()

  plt.xlim([time_steps[0], (future+5)*2])

  plt.xlabel('Time-Step')

  return plt
show_plot([x_train[0], y_train[0]], 0, 'Sample Example')
batch_size = 64

buffer_size = 10000



train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()



val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))

val_data = val_data.batch(batch_size).repeat()
print(x_train.shape[:])
simple_lstm_model = tf.keras.models.Sequential([

    tf.keras.layers.LSTM(100, input_shape=x_train.shape[-2:], return_sequences = True),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(100, return_sequences = True),

    tf.keras.layers.LSTM(100),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation="linear")

])



simple_lstm_model.compile(optimizer='adam', loss='mae')
for x, y in val_data.take(1):

    print(simple_lstm_model.predict(x).shape)
evaluation_interval = 100

epochs = 25



history = simple_lstm_model.fit(train_data, epochs=epochs,

                      steps_per_epoch=evaluation_interval,

                      validation_data=val_data, validation_steps=100)
print(simple_lstm_model.evaluate(x_val, y_val))
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
for x, y in val_data.take(10):

  plot = show_plot([x[0].numpy(), y[0].numpy(),

                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')

  plot.show()