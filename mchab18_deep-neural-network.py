# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Admission_Predict.csv')
data.head()
data.describe()
%matplotlib inline
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
train_set.head()
train_set.columns
train_X = train_set.drop(['Chance of Admit ','Serial No.'],axis=1)
train_y = train_set['Chance of Admit ']
corr_matrix = train_set.corr()
from pandas.plotting import scatter_matrix

attributes = ['GRE Score', "TOEFL Score", "Chance of Admit ",
              "CGPA"]
scatter_matrix(train_set[attributes], figsize=(12, 8))
from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
def build_model():
  model = keras.Sequential([
    layers.Dense(15, activation=tf.nn.elu, input_shape=[len(train_X.keys())]),
    layers.BatchNormalization(),
    layers.Dense(12, activation=tf.nn.elu),
    layers.BatchNormalization(),
    layers.Dense(10, activation=tf.nn.elu),
    layers.Dropout(.3),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
example_batch = train_X[:10]
example_result = model.predict(example_batch)
example_result
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 5000

history = model.fit(
  train_X, train_y,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
import matplotlib.pyplot as plt

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,1])

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,1])

plot_history(history)

test_X = test_set.drop(['Chance of Admit ','Serial No.'],axis=1)
test_y = test_set['Chance of Admit ']
loss, mae, mse = model.evaluate(test_X, test_y, verbose=0)

print('mae:',mae)
print('mse:',mse)
model.predict(test_X[:10])
test_y[:10]
test_predictions = model.predict(test_X).flatten()

plt.scatter(test_y, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - test_y
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
