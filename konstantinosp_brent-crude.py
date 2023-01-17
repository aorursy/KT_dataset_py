# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/brent-oil-prices/BrentOilPrices.csv")

df.head()
df.Date = pd.to_datetime(df.Date)

df.head()
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
TRAIN_SPLIT = int(0.8 * df.shape[0]) # selecting 80% as our training data 

uni_data = df.Price

uni_data

uni_data.plot()
uni_data = uni_data.values

uni_data
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()

uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

uni_data
univariate_past_history = 100

univariate_future_target = 0



x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,

                                           univariate_past_history,

                                           univariate_future_target)

x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,

                                       univariate_past_history,

                                       univariate_future_target)
def create_time_steps(length):

  time_steps = []

  for i in range(-length, 0, 1):

    time_steps.append(i)

  return time_steps
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
show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
BATCH_SIZE = 256

BUFFER_SIZE = 10000



train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))

train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))

val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
simple_lstm_model = tf.keras.models.Sequential([

    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),

    tf.keras.layers.Dense(1)

])



simple_lstm_model.compile(optimizer='adam', loss='mae')
EVALUATION_INTERVAL = 100

EPOCHS = 10



simple_lstm_model.fit(train_univariate, epochs=EPOCHS,

                      steps_per_epoch=EVALUATION_INTERVAL,

                      validation_data=val_univariate, validation_steps=50)
for x, y in val_univariate.take(10):

  plot = show_plot([x[0].numpy(), y[0].numpy(),

                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')

  plot.show()