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
import keras

import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
from keras.datasets import boston_housing



(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
import plotly.graph_objects as go
train_data[0]
train_data.shape
test_data.shape
def normalize(train_data, test_data):

    mean = train_data.mean(axis=0)

    std = train_data.std(axis=0)

    

    train_data = (train_data - mean) / std

    test_data = (test_data - mean) / std

    

    return train_data, test_data
train_data, test_data = normalize(train_data, test_data)
train_data[0]
from keras import layers, models
def build_model():

    model = models.Sequential()

    model.add(layers.Dense(128, activation='relu', input_shape=(train_data.shape[1], )))

    model.add(layers.Dense(16, activation='relu'))

    model.add(layers.Dense(1))

    

    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy',])

    return model
import numpy as np



k = 5

num_val_samples = len(train_data) // k

num_epochs = 500

all_mae_history = []

fig = go.FigureWidget()



for i in range(k):

    print('Processing fold #', i + 1)

    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]

    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    

    partial_train_data = np.concatenate([

            train_data[:i * num_val_samples],

            train_data[(i + 1) * num_val_samples:]],

            axis=0)

    partial_train_targets = np.concatenate([

            train_targets[:i * num_val_samples],

            train_targets[(i + 1) * num_val_samples:]],

            axis=0)

    

    model = build_model()

    history = model.fit(partial_train_data,

                        partial_train_targets,

                        epochs=num_epochs,

                        batch_size=1,

                        validation_data=(val_data, val_targets),

                        verbose=1)

    mae_history = history.history['val_mean_absolute_error']

    all_mae_history.append(mae_history)

    fig.add_scatter(y=mae_history)

    fig.show()
import tensorflow as tf


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 

sess = tf.Session(config=config) 

keras.backend.set_session(sess)
!pip install tensorflow-gpu