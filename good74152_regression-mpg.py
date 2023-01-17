#@title Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.
#@title MIT License

#

# Copyright (c) 2017 François Chollet

#

# Permission is hereby granted, free of charge, to any person obtaining a

# copy of this software and associated documentation files (the "Software"),

# to deal in the Software without restriction, including without limitation

# the rights to use, copy, modify, merge, publish, distribute, sublicense,

# and/or sell copies of the Software, and to permit persons to whom the

# Software is furnished to do so, subject to the following conditions:

#

# The above copyright notice and this permission notice shall be included in

# all copies or substantial portions of the Software.

#

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,

# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL

# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER

# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING

# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER

# DEALINGS IN THE SOFTWARE.
# Use seaborn for pairplot

!pip install seaborn
from __future__ import absolute_import, division, print_function, unicode_literals



import pathlib



import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



print(tf.__version__)
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

dataset_path
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',

                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path, names=column_names,

                      na_values = "?", comment='\t',

                      sep=" ", skipinitialspace=True)



dataset = raw_dataset.copy()

dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0

dataset['Europe'] = (origin == 2)*1.0

dataset['Japan'] = (origin == 3)*1.0

dataset.tail()
train_dataset = dataset.sample(frac=0.8,random_state=0)

test_dataset = dataset.drop(train_dataset.index)
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

plt.show()
train_stats = train_dataset.describe()

train_stats.pop("MPG")

train_stats = train_stats.transpose()

train_stats
train_labels = train_dataset.pop('MPG')

test_labels = test_dataset.pop('MPG')
def norm(x):

  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)

normed_test_data = norm(test_dataset)
def build_model():

  model = keras.Sequential([

    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(1)

  ])



  optimizer = tf.keras.optimizers.RMSprop(0.001)



  model.compile(loss='mean_squared_error',

                optimizer=optimizer,

                metrics=['mean_absolute_error', 'mean_squared_error'])

  return model
model = build_model()
model.summary()
example_batch = normed_train_data[:10]

example_result = model.predict(example_batch)

example_result
# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs):

    if epoch % 100 == 0: print('')

    print('.', end='')



EPOCHS = 1000



history = model.fit(

  normed_train_data, train_labels,

  epochs=EPOCHS, validation_split = 0.2, verbose=0,

  callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Abs Error [MPG]')

  plt.plot(hist['epoch'], hist['mean_absolute_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],

           label = 'Val Error')

  plt.ylim([0,5])

  plt.legend()



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Square Error [$MPG^2$]')

  plt.plot(hist['epoch'], hist['mean_squared_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_squared_error'],

           label = 'Val Error')

  plt.ylim([0,20])

  plt.legend()

  plt.show()





plot_history(history)
model = build_model()



# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,

                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])



plot_history(history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)



print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
test_predictions = model.predict(normed_test_data).flatten()



plt.scatter(test_labels, test_predictions)

plt.xlabel('True Values [MPG]')

plt.ylabel('Predictions [MPG]')

plt.axis('equal')

plt.axis('square')

plt.xlim([0,plt.xlim()[1]])

plt.ylim([0,plt.ylim()[1]])

_ = plt.plot([-100, 100], [-100, 100])

plt.show()

error = test_predictions - test_labels

plt.hist(error, bins = 25)

plt.xlabel("Prediction Error [MPG]")

_ = plt.ylabel("Count")

plt.show()