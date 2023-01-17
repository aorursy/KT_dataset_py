import tensorflow as tf

import numpy as np

import pandas as pd
input_columns = ['year', 'month', 'day', 'hour', 'minute', 'moisture0', 'moisture1','moisture2']

prediction = ['moisture3']
train_size = 3000

test_size = 1000
data_frame = pd.read_csv('../input/soil-moisture-dataset/plant_vase1.CSV')
data_frame
data_frame.columns
data_frame.drop(columns=['irrgation'])
data_frame[['moisture0', 'moisture1','moisture2', 'moisture3']].plot()
X_train = data_frame[input_columns][:train_size]

Y_train = data_frame[prediction][:train_size]

X_test = data_frame[input_columns][train_size:]

Y_test = data_frame[prediction][train_size:]
X_train.shape, Y_train.shape
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=1, input_shape=(len(input_columns),)))

model.summary()
model.compile(optimizer='adam', loss='MSE', metrics=['mean_absolute_error'])
model.fit(X_train,Y_train, epochs=100)
model.evaluate(X_test, Y_test)
model.metrics_names