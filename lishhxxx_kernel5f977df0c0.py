import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

print(tf.__version__)
raw_dataset = pd.read_csv('./bike-rental-prediction/train.csv')
dataset = raw_dataset.copy()
dataset.tail()
# transforming dteday to day month and year
dataset['day'] = pd.DatetimeIndex(dataset['dteday']).day
dataset['month'] = pd.DatetimeIndex(dataset['dteday']).month
dataset['year'] = pd.DatetimeIndex(dataset['dteday']).year

# removing not important columns
dataset.drop(columns = ['instant','dteday', 'mnth', 'casual', 'registered'],inplace=True)
dataset.tail()
# splitting to testing and training datasets
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats.pop("cnt")
train_stats = train_stats.transpose()
train_stats
train_labels = train_dataset.pop('cnt')
test_labels = test_dataset.pop('cnt')
# normalization of dataset
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_train_data.head()
from tensorflow.keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def rmsle(y, y0):
        return K.sqrt(K.mean(K.square(tf.math.log1p(y) - tf.math.log1p(y0))))
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, kernel_constraint=tf.keras.constraints.NonNeg())
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss=root_mean_squared_error,
                optimizer=optimizer,
                metrics=[root_mean_squared_error, rmsle,'mae', 'mse'])

    return model
model = build_model()
model.summary()
EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])
# history of loss and metrics
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "loss")
loss, root_mean_squared_error, rmsle, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('CNT')
plt.ylabel('CNT Predictions')
plt.plot()
test_labels[0:10]
test_predictions[0:10]
raw_test_dataset = pd.read_csv('./bike-rental-prediction/test.csv')
original_test_dataset = raw_test_dataset.copy()
# removing not important columns
original_test_dataset['day'] = pd.DatetimeIndex(original_test_dataset['dteday']).day
original_test_dataset['month'] = pd.DatetimeIndex(original_test_dataset['dteday']).month
original_test_dataset['year'] = pd.DatetimeIndex(original_test_dataset['dteday']).year

original_test_dataset_instant  = original_test_dataset.pop('instant')
original_test_dataset.drop(columns = ['dteday', 'mnth'],inplace=True)

original_test_dataset.tail()
normed_original_test_data = norm(original_test_dataset)
normed_original_test_data.head()
original_test_predictions = model.predict(normed_original_test_data).flatten()
original_test_predictions
len(original_test_predictions)
result_file = open('./results/r2.txt', 'w')
result_file.write("instant,cnt\n")

for i in range(0, len(original_test_dataset_instant) - 1):
    result_file.write(str(original_test_dataset_instant[i]) + "," + str(original_test_predictions[i]) + "\n")
    
result_file.close()


