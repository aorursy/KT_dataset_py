# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
x = np.array([-40, -30, -20, -10,  0,  8, 15, 22, 38, ],  dtype=float)

y= np.array([-40, -22, -4, 14, 32, 46.4, 59, 71.6, 100.4],  dtype=float)
layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer_0])
model.compile(loss="mean_squared_error",

              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(x, y, epochs=500, verbose=False)
model.predict([47])
print("These are the layer variables: {}".format(layer_0.get_weights()))
    