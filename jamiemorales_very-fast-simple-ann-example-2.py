# Set-up libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
x = np.array([7, 1, 3, 9, 9, 8, 2, 2, 4, 4], dtype=float)
y = np.array([13,  1,  5, 17, 17, 15,  3,  3,  7,  7], dtype=float)
# Build and train neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(x, y, epochs=500)
# Apply the neural network
model.predict([11.0])
