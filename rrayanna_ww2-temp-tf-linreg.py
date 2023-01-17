import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow import feature_column

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from sklearn import metrics
data = pd.read_csv('../input/weatherww2/Summary of Weather.csv')

data.head()
data.info()
sns.pairplot(data[['MinTemp','MaxTemp']]) # Strong Correlation Between Min and Max Temps
data_new = data[['MinTemp', 'MaxTemp']] # Select columns needed
# Prep Train and Test Data

train_data = data_new.sample(frac=0.8,random_state=42) 

test_data = data_new.drop(train_data.index)
# Get Stats on Train Data

train_stats = train_data.describe()

train_stats.pop("MaxTemp")

train_stats = train_stats.transpose()

train_stats
# Create Labels needed for TF model

train_labels = train_data.pop('MaxTemp')

test_labels = test_data.pop('MaxTemp')
def norm(x):

  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_data)

normed_test_data = norm(test_data)
def build_model():

  model = tf.keras.Sequential([

    layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),

    layers.Dense(64, activation='relu'),

    layers.Dense(1)

  ])



  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',optimizer=optimizer, metrics=['mae', 'mse'])



  return model
model = build_model()

model.summary()
# Check an example

example_batch = normed_train_data[:10]

example_result = model.predict(example_batch)

example_result
# Train the data

# Added early stopping to avoid overfitting



epochs = 25

history = model.fit(normed_train_data, train_labels, epochs=epochs, validation_split = 0.2, 

                    verbose=0, 

                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
test_predictions = model.predict(normed_test_data).flatten()



a = plt.axes(aspect='equal')

plt.scatter(test_labels, test_predictions)

plt.xlabel('True Values [MaxTemp]')

plt.ylabel('Predictions [MaxTemp]')

lims = [0, 50]

plt.xlim(lims)

plt.ylim(lims)

_ = plt.plot(lims, lims)
error = test_predictions - test_labels

plt.hist(error, bins = 25)

plt.xlabel("Prediction Error [Max Temp]")

_ = plt.ylabel("Count")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, test_predictions)))