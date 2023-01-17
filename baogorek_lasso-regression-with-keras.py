%env KERAS_BACKEND=theano
import theano

import numpy as np
from numpy.random import normal
from numpy.random import seed
import matplotlib.pyplot as plt

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l1, l2
# Simulating the data
seed(44234)
N = 1000

x = normal(size = N).astype(np.float32).reshape((N, 1))
z = normal(size = N).astype(np.float32).reshape((N, 1))
w = normal(size = N).astype(np.float32).reshape((N, 1))
X = np.concatenate((x, w, z), axis = 1)
epsilon = normal(size = N, scale = 7.5)

y = 1.1 + 3.2 * X[:, 0] + epsilon
y = y.astype(np.float32).reshape((N, 1))

plt.scatter(X[:, 0], y[:, 0])
plt.title("y vs x_1: positive relationship")
plt.show()

plt.scatter(X[:, 1], y[:, 0])
plt.title("y vs x_2: no relationship")
plt.show()

plt.scatter(X[:, 2], y[:, 0])
plt.title("y vs x_3: no relationship")
plt.show()
input_x = Input(shape = (3,))
lin_fn = Dense(1)(input_x)
yx_model = Model(inputs = input_x, outputs = lin_fn)
yx_model.compile(loss = 'mean_squared_error', optimizer = "sgd")
yx_model.fit(X, y, epochs = 10000, batch_size = 500, verbose = 0)
weights = yx_model.get_weights()
b = weights[0]
print("Coefficients for x_1, x_2, x_3 are %.3f, %.3f, %.3f, respectively" %
      (b[0], b[1], b[2]))
lambda_val = .7
lasso_input = Input(shape = (3,))
lasso_fn = Dense(1, kernel_regularizer = l1(lambda_val))(lasso_input)
lasso_model = Model(inputs = lasso_input, outputs = lasso_fn)
lasso_model.compile(loss = 'mean_squared_error', optimizer = "sgd")
lasso_model.fit(X, y, epochs = 10000, batch_size = 500, verbose = 0)
weights = lasso_model.get_weights()
b = weights[0]
print("Coefficients for x_1, x_2, x_3 are %.3f, %.3f, %.3f, respectively" %
      (b[0], b[1], b[2]))