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
        
!pip install -q git+https://github.com/tensorflow/docs

import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

DIVIDER = 100000

df = pd.read_csv('/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')

CATEGORICAL_COLUMNS = ['Port Name', 'State', 'Border', 'Date', 'Measure']

NUMERIC_COLUMNS = ['Port Code']

train_label = df['Value']
train_data = df.drop('Value', axis=1)

for feature_name in CATEGORICAL_COLUMNS:
  dist_labels = []

  vocabulary = train_data[feature_name].unique()
  for item in vocabulary:
    dist_labels.append(item)

  train_data[feature_name] = train_data[feature_name].map(lambda x: dist_labels.index(x))

train_stats = train_data.describe().transpose()

def norm_train(x):
  return (x - train_stats['mean']) / train_stats['std']

train_data = norm_train(train_data)
train_label = train_label / DIVIDER
def build_model():
  model = keras.Sequential([
    layers.Dense(128, activation='relu', input_dim=train_data.shape[1]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

model.summary()

x_train,x_test,y_train,y_test = train_test_split(train_data, train_label, test_size=0.20)

EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(
  x_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, tfdocs.modeling.EpochDots()])


test_predictions = model.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [COUNTS]')
plt.ylabel('Predictions [COUNTS]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()

error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [COUNTS]")
_ = plt.ylabel("Count")

plt.show()

norm_error = (test_predictions - y_test) * DIVIDER
print(norm_error.describe())