# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals



from tensorflow import keras



import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

from itertools import cycle, islice

import json



from tensorflow import feature_column

from sklearn.model_selection import train_test_split



from tensorflow.keras import datasets, layers, models, Sequential



def load_beat_data(filename):

    with open(filename) as file:

        return np.array(json.load(file))





def load_train_data():    

    john_data = load_beat_data('../input/john.json')

    simon_data = load_beat_data('../input/simon.json')    

    

    features = np.concatenate([john_data, simon_data])

    

    labels = np.concatenate([np.zeros(len(john_data)), np.zeros(len(simon_data)) + 1])

    

    return train_test_split(features, labels, test_size=0.1, random_state=42)



train_data, test_data, train_labels, test_labels = load_train_data()



train_data = (train_data - train_data.mean())/train_data.std()

test_data = (test_data - test_data.mean())/test_data.std()





model = Sequential([

    layers.Reshape((20, 1), input_shape=(20,)),

    layers.Conv1D(40, 8, activation = 'relu', input_shape = (20, 1)),

    layers.Conv1D(20, 4, activation = 'relu'),

    layers.MaxPooling1D(1),

    layers.Conv1D(60, 4, activation = 'relu'),

    #layers.GlobalAveragePooling1D(),

    layers.Flatten(),

    layers.Dense(16, activation = 'relu'),

    layers.Dense(2, activation='softmax')

])

"""

model = Sequential([

    layers.Flatten(input_shape=(21, )),

    layers.Dense(16, activation='relu'),

    layers.Dense(32, activation='relu'),

    layers.Dense(16, activation='relu'),

    layers.Dense(2, activation='softmax')

])

"""



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(train_data, train_labels, epochs=60)



test_loss, test_acc = model.evaluate(test_data, test_labels)

print('\nTest accuracy:', test_acc)
