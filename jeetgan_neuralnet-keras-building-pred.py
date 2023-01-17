# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from matplotlib import pyplot as plt

import os

cwd = os.getcwd()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



test_filepath = '/kaggle/input/UjiIndoorLoc/ValidationData.csv'

train_filepath = '/kaggle/input/UjiIndoorLoc/TrainingData.csv'

# Any results you write to the current directory are saved as output.
training_data = pd.read_csv(train_filepath)

test_data = pd.read_csv(test_filepath)

train_building_X = train_lat_long_X

train_building_Y = training_data.loc[:, "BUILDINGID"]

test_building_X = test_lat_long_X

test_building_Y = test_data.loc[:, "BUILDINGID"]

val_data_building = (test_building_X, test_building_Y)
def get_accuracy(predictions, actual_values):

  correct_classif = [1 if pred_i == actual_i else 0 for pred_i, actual_i in zip(predictions, actual_values)]

  correct_classif = np.array(correct_classif)

  return correct_classif.sum() / len(actual_values)
model_bldn_1 = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(724, activation=tf.nn.relu),

  tf.keras.layers.Dense(3, activation = 'softmax')

])

model_bldn_1.compile(optimizer='adam', 

                loss='sparse_categorical_crossentropy',

                metrics = ['accuracy'])

epochs = 5

result_2 = model_bldn_1.fit(train_building_X.values.astype(float), train_building_Y.values.astype(float), validation_data = val_data_building, epochs=epochs)
plt.plot(result_2.history['loss'])
plt.plot(result_2.history['val_loss'])