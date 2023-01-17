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
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)

std = train_data.std(axis=0)

train_data = (train_data-mean)/std

test_data = (test_data-mean)/std
from keras import models, layers

def build_model():

    model = models.Sequential()

    model.add(layers.Dense(64, activation = 'relu', input_shape = (train_data.shape[1], )))

    model.add(layers.Dense(64, activation = 'relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'] )

    return model
k=4

num_val_samples = len(train_data) // k

num_epochs = 100

all_mae_histories = []



for i in range(k):

    print('Processing fold # ', i)

    x_val = train_data[i*num_val_samples : (i+1)*num_val_samples]

    y_val = train_targets[i*num_val_samples : (i+1)*num_val_samples]

    x_train = np.concatenate([train_data[:i*num_val_samples] ,train_data[(i+1)*num_val_samples:]], axis=0)

    y_train = np.concatenate([train_targets[:i*num_val_samples] ,train_targets[(i+1)*num_val_samples:]], axis=0)

    print(x_train.shape, y_train.shape)

    model = build_model()

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=1, verbose=0)

    mae_hist = history.history['val_mean_absolute_error']

    all_mae_histories.append(mae_hist)

    
average_mae_history = [

np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
average_mae_history
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()
def smooth_curve(points, factor=0.85):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous * factor + point * (1 - factor))

        else:

            smoothed_points.append(point)

    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()
model = build_model()

model.fit(train_data, train_targets,

epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score