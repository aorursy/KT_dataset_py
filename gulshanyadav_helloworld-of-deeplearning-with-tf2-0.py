import numpy as np # linear algebra

import tensorflow as tf

from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
x_train = np.array([-2, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5, 6, 7, 8], dtype=float)

y_train = np.array([6.0, 8.0, 10.0, 11.0, 14.0, 16.0, 18.0, 20, 22, 24, 26], dtype=float)
model.fit(x_train, y_train, epochs=500)
print(model.predict([30.0]))