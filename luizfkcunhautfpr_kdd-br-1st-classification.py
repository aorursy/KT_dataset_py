import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plots
import graphviz # print tree
from sklearn import datasets, tree

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
X = pd.read_csv("../input/train.csv", usecols=['field','age','type', 'harvest_month']).values # harvest year foi retirado devido a análise 2
y = pd.read_csv("../input/train.csv", usecols=['production']).values
#X.head() #nao e possivel usar com o .values
#y.head() #nao e possivel usar com o .values
y = y.flatten()
tamanhoteste = len(X)/5
X_train = X[:int(4*tamanhoteste)]
y_train = y[:int(4*tamanhoteste)]
X_test = X[int(1*tamanhoteste):]
y_test = y[int(1*tamanhoteste):]
print(X_train.shape, y_train.shape)
#criação do model
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model
model = build_model()
model.summary()
class PrintDot(keras.callbacks.Callback): # para monitoramnento do progresso
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000)

history = model.fit(X_train, y_train, epochs=2000,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
plot_history(history)
[loss, mae] = model.evaluate(X_test, y_test, verbose=1)
print("Testing set Mean Abs Error: ", (mae))
test_predictions = model.predict(X_test).flatten()
print(test_predictions.shape)
print(y_test.shape)
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
