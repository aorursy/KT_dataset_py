# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# libraries

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib as mpl

import matplotlib.pyplot as plt

tf.random.set_seed(13)
comfirm_data = pd.read_csv("../input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv")

comfirm_data.head()
train_data = pd.DataFrame(comfirm_data.iloc[:,4:].sum(axis=0), columns = ["number"])

plot_range = range(len(train_data))

plt.bar(plot_range,train_data["number"],label='Global')

plt.legend()

plt.title('Total Number of COVID-19 confirm in Global')

plt.xlabel('Day')

plt.ylabel('Number of confirmed')

plt.grid()

plt.show()
from sklearn.preprocessing import MinMaxScaler

all_data = pd.DataFrame(data = train_data)

scaler_features = MinMaxScaler(feature_range=(0, 1))

scaler_labels = MinMaxScaler(feature_range=(0, 1))



TRAIN_SPLIT = len(all_data) - 14



dataset = all_data.values

scaler_features.fit(dataset[:TRAIN_SPLIT])

scaler_labels.fit(dataset[:TRAIN_SPLIT].reshape(-1,1))

std_data = scaler_features.transform(dataset)
def create_data(dataset, start_index, end_index, history_size, target_size):

    data = []

    labels = []

    

    start_index = start_index + history_size

    if end_index is None:

        end_index = len(dataset) - target_size

    

    for i in range(start_index, end_index):

        indices = range(i-history_size,i)

        data.append(np.reshape(dataset[indices],(history_size,1)))

        labels.append(dataset[i+target_size])

    return np.array(data,dtype=np.float),np.array(labels,dtype=np.float)

past_history = 5

future_target = 0



x_train, y_train = create_data(std_data, 38, TRAIN_SPLIT, past_history, future_target)

x_val, y_val = create_data(std_data, TRAIN_SPLIT, None, past_history, future_target)
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

  plt.xlim([time_steps[0], (future+1)])

  plt.xlabel('Time-Step')

  return plt
BATCH_SIZE = 256

BUFFER_SIZE = 10000



train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))

val_data = val_data.batch(BATCH_SIZE).repeat()
simple_lstm_model = tf.keras.models.Sequential([

    tf.keras.layers.LSTM(8, input_shape=x_train.shape[-2:]),

    tf.keras.layers.Dense(1)

])



simple_lstm_model.compile(optimizer='adam', loss='mae')
EVALUATION_INTERVAL = 200

EPOCHS = 13

simple_lstm_model.fit(train_data, epochs=EPOCHS,

                      steps_per_epoch=EVALUATION_INTERVAL,

                      validation_data=val_data, validation_steps=50)
for x, y in val_data.take(1):

  plot = show_plot([x[0].numpy(), y[0].numpy(),

                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')

  plot.show()
day = 14

result = []

for i in range(day):

    data_pred = std_data[TRAIN_SPLIT-past_history+i+1:TRAIN_SPLIT+i+1]

    x = data_pred.reshape(1,5,1)

    pred = simple_lstm_model.predict(x)

    result.append(pred[0])

    

temp = list(std_data[0:TRAIN_SPLIT])

temp += result
train_data = pd.DataFrame(comfirm_data.iloc[:,4:].sum(axis=0), columns = ["number"])

pred_data = scaler_features.inverse_transform(temp)

plt.figure(figsize=(10,8))

plot_range = range(len(train_data))

plt.bar(plot_range,train_data["number"],label='Global')

plt.plot(pred_data, label='Predicte',color='orange')

plt.legend()

plt.title('Total Number of COVID-19 confirm in Global')

plt.xlabel('Day')

plt.ylabel('Number of Confirmed')

plt.grid()

plt.show()