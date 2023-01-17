import tensorflow as tf

import numpy as np
tf.__version__
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)

fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
model = tf.keras.Sequential([

    tf.keras.layers.Dense(1, input_shape=[1])

])
model.compile(optimizer=tf.keras.optimizers.Adam(0.1),

             loss=tf.keras.losses.mean_squared_error)
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.plot(history.history['loss'])

plt.show()
model.predict([100])
model.get_layer(index=0).get_weights()
model = tf.keras.Sequential([

    tf.keras.layers.Dense(4, input_shape=[1]),

    tf.keras.layers.Dense(4),

    tf.keras.layers.Dense(1),

])
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss="mean_squared_error")
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.plot(history.history['loss'])

plt.show()
model.predict([100])