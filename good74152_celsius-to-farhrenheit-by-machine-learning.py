#@title Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf



import numpy as np
import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)

fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)



for i,c in enumerate(celsius_q):

  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])  
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',

              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

print("Finished training the model")
import matplotlib.pyplot as plt

plt.xlabel('Epoch Number')

plt.ylabel("Loss Magnitude")

plt.plot(history.history['loss'])
print(model.predict([100.0]))
print("These are the layer variables: {}".format(l0.get_weights()))
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])  

l1 = tf.keras.layers.Dense(units=4)  

l2 = tf.keras.layers.Dense(units=1)  

model = tf.keras.Sequential([l0, l1, l2])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

print("Finished training the model")

print(model.predict([100.0]))

print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))

print("These are the l0 variables: {}".format(l0.get_weights()))

print("These are the l1 variables: {}".format(l1.get_weights()))

print("These are the l2 variables: {}".format(l2.get_weights()))