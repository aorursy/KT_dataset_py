from tensorflow.python.client import device_lib
import matplotlib as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

season_stats = pd.read_csv('../input/nba-players-stats/Seasons_Stats.csv', header=0)
season_stats['PPG'] = season_stats['PTS'] / season_stats['G']

categories = ['PPG', '3PAr', '2P%', 'eFG%', 'FT%', '3P%']

for s in categories:
    season_stats[s] = season_stats[s].fillna(0.00)

recent_stats = season_stats[season_stats['Year'] >= 2005]

played_stats = recent_stats[recent_stats['G'] > 10]

prime_stats = played_stats[abs(played_stats['Age'] - 27) <= 3]

complete_set = recent_stats[categories]

train_dataset = complete_set.sample(frac=0.8, random_state=0)
test_dataset = complete_set.drop(train_dataset.index)

train_labels = train_dataset.pop('3P%')
test_labels = test_dataset.pop('3P%')

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[
            len(train_dataset.keys())]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.0001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model


model = build_model()

epochs = 5

print(train_dataset)

history = model.fit(
    train_dataset, train_labels,
    epochs=epochs, validation_split=0.2, verbose=2)

loss = model.evaluate(test_dataset, test_labels, verbose=1)

print("Eval loss: {}".format(loss))
# Predictions

curry = np.array([[27.3, 0.603, 0.525, 0.604, 0.916]])

curry_predict = model.predict(curry)

print("Curry {}".format(curry_predict))

simmons = np.array([[16.7, 0, 0.588, 0.587, 0.627]])

simmons_predict = model.predict(simmons)

print("Simmons {}".format(simmons_predict))