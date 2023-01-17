import numpy as np

from scipy.stats.mstats import gmean, hmean, skew

import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error
N_SAMPLES = 10000

N_TEST = 1000

MAX_TIMESTEPS = 8

MASK_VALUE = -1



train_X = np.random.uniform(size = (N_SAMPLES, MAX_TIMESTEPS, 1))

train_L = np.random.randint(1, MAX_TIMESTEPS, N_SAMPLES)



test_X = np.random.uniform(size = (N_TEST, MAX_TIMESTEPS, 1))

test_L = np.random.randint(1, MAX_TIMESTEPS, N_TEST)
for i in range(N_SAMPLES):

    train_X[i, train_L[i]] = MASK_VALUE
for i in range(N_TEST):

    train_X[i, test_L[i]] = MASK_VALUE
train_y = skew(train_X, axis = 1)

test_y = skew(test_X, axis = 1)
input_ = tf.keras.Input(shape = (None, 1))

masked = tf.keras.layers.Masking(MASK_VALUE)(input_)

lstm1 = tf.keras.layers.LSTM(1, return_sequences = True)(masked)

lstm2 = tf.keras.layers.LSTM(1)(lstm1)

output = tf.keras.layers.Dense(1)(lstm2)

model = tf.keras.Model(inputs = input_, outputs = output)

model.summary()
model.compile('adam', 'mse')
hist = model.fit(train_X, train_y, epochs = 1)
prediction = model.predict(test_X)
prediction[:5]
test_y[:5]
mean_absolute_error(test_y, prediction)
mean_squared_error(test_y, prediction)
np.percentile(test_y, np.arange(0, 100, 10))