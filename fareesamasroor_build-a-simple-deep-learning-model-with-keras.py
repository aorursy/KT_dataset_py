import tensorflow as tf

import numpy as np

from tensorflow import keras



model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

y = np.array([-4.0, -1.0, 2.0, 5.0, 8.0, 11.0,14.0 ], dtype=float)

model.fit(x, y, epochs=500)
model.predict([13.0])