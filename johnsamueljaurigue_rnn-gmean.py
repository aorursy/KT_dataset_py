import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from scipy.stats import gmean
N_SAMPLES = 100000
N_TEST = 1000
MAX_TIMESTEPS = 9
MASK_VALUE = -1

train_X = np.random.uniform(size=(N_SAMPLES, MAX_TIMESTEPS, 1))
train_L = np.random.randint(4, MAX_TIMESTEPS, N_SAMPLES)

test_X = np.random.uniform(size=(N_TEST, MAX_TIMESTEPS, 1))
test_L = np.random.randint(4, MAX_TIMESTEPS, N_TEST)
for i in range(N_SAMPLES):
    train_X[i, train_L[i]:] = MASK_VALUE
for i in range(N_TEST):
    test_X[i, test_L[i]:] = MASK_VALUE
train_y = np.ma.masked_array(train_X, train_X==MASK_VALUE).std(axis=1).data
test_y = np.ma.masked_array(test_X, test_X==MASK_VALUE).std(axis=1).data
input_ = tf.keras.Input(shape=(None, 1))
masked = tf.keras.layers.Masking(MASK_VALUE)(input_)
lstm1 = tf.keras.layers.LSTM(32, return_sequences=True)(masked)
lstm2 = tf.keras.layers.LSTM(32)(lstm1)
output = tf.keras.layers.Dense(1)(lstm2)

model = tf.keras.Model(inputs=input_, outputs=output)
model.summary()
model.compile('adam', 'mse')
hist = model.fit(train_X, train_y, epochs=3)
prediction = model.predict(test_X)
prediction[:5]
test_y[:5]
mse = model.evaluate(train_X,train_y)
gmean_ = gmean(train_y,axis = 0)
# Value of Geometric Mean
gmean_
# Value of Mean Squared Error
mse
# Value of Mean absolute error
mean_absolute_error(test_y, prediction)
# Value of Decile
np.percentile(train_X,10)