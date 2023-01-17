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



loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()

model.compile(loss= loss_fn, optimizer='adam', metrics=[loss_fn])

model.fit(x_train.values, y_train.values, epochs=15, batch_size=64, validation_split = 0.2, verbose =0)
'''Predict the training data'''

predictions = model.predict(x_train.values).squeeze()
def mean_absolute_error(y_train, predictions):

    """

    Returns Mean Absolute Error 

    """

    abs_error = abs(y_train - predictions)

    mean_abs_error = abs_error.mean()

    return mean_abs_error



mean_absolute_error(y_train, predictions)
def mean_squared_error(y_train, predictions):

    """

    Returns Mean Squared Error 

    """

    square_error = (y_train - predictions) ** 2

    mean_square_error = square_error.mean()

    return mean_square_error



mean_squared_error(y_train, predictions)
def root_mean_squared_error(y_train, predictions):

    """

    Returns Root Mean Squared Error

    """

    square_error = (y_train - predictions) ** 2

    mean_square_error = square_error.mean()

    rmse = np.sqrt(mean_square_error)

    return rmse



root_mean_squared_error(y_train, predictions)
def root_mean_squared_log_error(y_train, predictions):

    """

    Returns Root Mean Square Log Error

    """

    square_error = (np.log(y_train+1) - np.log(predictions+1)) ** 2

    mean_square_log_error = square_error.mean()

    rmsle = np.sqrt(mean_square_log_error)

    return rmsle



root_mean_squared_log_error(y_train, predictions)
"""Use Stochastic Gradient Descent"""

optimizer = tf.keras.optimizers.SGD()

model.compile(loss= loss_fn, optimizer= optimizer, metrics=[loss_fn])

model.fit(x_train.values, y_train.values, epochs=15, batch_size=64, validation_split = 0.2, verbose = 0)
optimizer = tf.keras.optimizers.RMSprop()

model.compile(loss= loss_fn, optimizer= optimizer, metrics=[loss_fn])

model.fit(x_train.values, y_train.values, epochs=15, batch_size=64, validation_split = 0.2, verbose = 0)
optimizer = tf.keras.optimizers.Adam()

model.compile(loss= loss_fn, optimizer= optimizer, metrics=[loss_fn])

model.fit(x_train.values, y_train.values, epochs=15, batch_size=64, validation_split = 0.2, verbose = 0)