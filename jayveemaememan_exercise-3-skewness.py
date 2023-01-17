from scipy.stats.mstats import skew 

import numpy as np

import tensorflow as tf

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error
#set the unused parts of the data as -1.

N_SAMPLES = 10000

N_TEST = 1000

#N=5

#S=(5%5)+1=5

MAX_TIMESTEPS = 6

MASK_VALUE = -1



train_X = np.random.uniform(size=(N_SAMPLES, MAX_TIMESTEPS, 1))

train_L = np.random.randint(2, MAX_TIMESTEPS, N_SAMPLES)



test_X = np.random.uniform(size=(N_TEST, MAX_TIMESTEPS, 1))

test_L = np.random.randint(2, MAX_TIMESTEPS, N_TEST)
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
hist = model.fit(train_X, train_y, epochs=5)
prediction = model.predict(test_X)
prediction[:5]
test_y[:5]
#Finding skewness

x = test_y 

  

print (x) 



  

print('\nSkewness for data : ', skew(x, axis=0, bias=True))

#Finding mean absolute error

mean_absolute_error(test_y, prediction)
#Finding mean squared error

mean_squared_error(test_y, prediction)
#Finding Decile error

np.percentile(x, np.arange(0, 100, 10)) 