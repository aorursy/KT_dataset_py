# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf



import numpy as np

import matplotlib.pyplot as plt
import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)
celsius_q    = np.array([0,  8, 15, 22,  38],  dtype=float)

fahrenheit_a = np.array([32, 46.4, 59, 71.6, 100.4],  dtype=float)



for i,c in enumerate(celsius_q):

  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=550, verbose=False)

plt.xlabel('Epoch Number')

plt.ylabel('Loss magnitude')

plt.plot(history.history['loss'])

print(model.predict([100]))
print(l0.get_weights())
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])

l1 = tf.keras.layers.Dense(units=4)

l2 = tf.keras.layers.Dense(units=1)

model = tf.keras.models.Sequential([l0, l1, l2])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

model.fit(celsius_q, fahrenheit_a, epochs = 500, verbose=False)
model.predict([100])
print("The lo variables weight {}".format(l0.get_weights()))

print("The l1 variables weight {}".format(l1.get_weights()))

print("The l2 variables weight {}".format(l2.get_weights()))
