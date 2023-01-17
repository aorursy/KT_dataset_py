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
import keras

from keras.datasets import boston_housing

from keras import models

from keras import layers

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import matplotlib.pyplot as plt



(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)

print(test_data.shape)
# Centering and scaling data



mean= train_data.mean(axis = 0)

train_data -= mean

std = train_data.std(axis = 0)

train_data /= std



test_data -= mean

test_data /= std
# Model Build



def build_model():

    model = models.Sequential()

    model.add(layers.Dense(64,activation='relu', input_shape=(train_data.shape[1],)))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model
# K - Fold cross validation



import numpy as np



k=2

num_val_samples = len(train_data) //k

num_epochs = 100

all_scores = []
num_epochs = 500

all_mae_histories = []

for i in range(k):

    print('processing fold #',i)

    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]

    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    

    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)

    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)

    

    model = build_model()

    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data,val_targets), epochs=num_epochs, batch_size=16,verbose=0)

    mae_history = history.history['val_mean_absolute_error']

    all_mae_histories.append(mae_history)



average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# estimate accuracy on whole dataset using loaded weights

scores = model.evaluate(val_data, val_targets, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
def smooth_curve(points, factor=0.9):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous*factor + point*(1-factor))

        else:

            smoothed_points.append(point)

    return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:])



plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()
# Model Build



def build_model_dropout():

    model = models.Sequential()

    model.add(layers.Dense(64,activation='relu', input_shape=(train_data.shape[1],)))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model
num_epochs = 500

all_mae_histories = []

for i in range(k):

    print('processing fold #',i)

    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]

    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    

    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)

    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)

    

    model = build_model_dropout()

    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data,val_targets), epochs=num_epochs, batch_size=16,verbose=0)

    mae_history = history.history['val_mean_absolute_error']

    all_mae_histories.append(mae_history)



average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# estimate accuracy on whole dataset using loaded weights

scores = model.evaluate(val_data, val_targets, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
smooth_mae_history = smooth_curve(average_mae_history[10:])



plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()
num_epochs = 5000

all_mae_histories = []

for i in range(k):

    print('processing fold #',i)

    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]

    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    

    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)

    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)

    

    model = build_model_dropout()

    

    # checkpoint

    filepath="weights.best.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.9,patience=150, min_lr=0.001,  verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)

    callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]

    

    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data,val_targets), epochs=num_epochs, 

                        callbacks=callbacks_list, batch_size=16,verbose=0)

    mae_history = history.history['val_mean_absolute_error']

    all_mae_histories.append(mae_history)
model.load_weights("weights.best.hdf5")
# estimate accuracy on whole dataset using loaded weights

scores = model.evaluate(val_data, val_targets, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))