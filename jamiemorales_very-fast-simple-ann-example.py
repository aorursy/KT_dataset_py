# Set-up libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Manufacture data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
# Build and train neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile  neural network
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the  neural network
model.fit(xs, ys, epochs=500)
# Apply the neural network
model.predict([10.0])
