import numpy as np

import pandas as pd 

import tensorflow as tf

pd.set_option('max.columns',80)



preprocessed_train = pd.read_csv('../input/preprocessed-train-data/preprocessed_train_data.csv')

test = pd.read_csv('../input/preprocessed-test-data/preprocessed_test_data.csv')

display(preprocessed_train.head())

'''Split the train data into features and target'''

x_train, y_train = preprocessed_train[preprocessed_train.columns[:-1]], preprocessed_train[preprocessed_train.columns[-1]]
'''Kernel initializer denotes the distribution in which the weights of the neural networks are initialized'''

model = tf.keras.Sequential([

    tf.keras.layers.Dense(128, kernel_initializer='normal', activation='relu'),

    tf.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dense(384, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dense(384, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dense(1, kernel_initializer='normal',activation='linear')

])



msle = tf.keras.losses.MeanSquaredLogarithmicError()

model.compile(loss= msle, optimizer='adam', metrics=[msle])

model.fit(x_train.values, y_train.values, epochs=15, batch_size=64, validation_split = 0.2)
predictions = model.predict(test.values)

predictions