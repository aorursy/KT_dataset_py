import tensorflow as  tf

import numpy as np

from tensorflow import keras
print("tensorflow verion :",tf.__version__)
print("Eager mode enabled :",tf.executing_eagerly())
print("GPU", "Available" if tf.config.list_physical_devices('GPU') else "Notavailable")
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
X_train = np.array([2.50, 3.55, 4.50, 5.60, 6.99, 7.23, 8.45, -2.50, -3.55, -4.50, -5.60, -6.99, -7.23, -8.45,], dtype = float)

Y_train = np.array([14.50, 19.75, 24.50, 30.00, 36.95, 38.15, 44.25, -10.50, -15.75, -20.50, -26.00, -32.95, -34.15, -40.25], dtype = float)
import matplotlib.pyplot as plt

plt.scatter(X_train, Y_train, color = 'blue')

plt.show()
from statistics import mean

def prediction (x):

  m = (mean(X_train)*mean(Y_train)-mean(X_train*Y_train))/(mean(X_train)*mean(X_train)-mean(X_train*X_train))

  b = mean(Y_train) - m*mean(X_train)

  y = m *x +b

  return y
prediction(10)
model.fit(X_train, Y_train, epochs=500)
model.predict([10.00])
model.get_weights()