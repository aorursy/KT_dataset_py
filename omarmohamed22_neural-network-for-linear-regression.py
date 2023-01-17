# This is a Fahrenheit celsuis machine learning model to predict the relation between them :



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from keras import layers,Sequential,optimizers,losses

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
celsius_q    = [-40, -10,  0,  8, 15, 22,  38 , 32  , 60 ,122  ,1 ]

fahrenheit_a = [-40,  14, 32, 46, 59, 72, 100 , 89.6, 140,251.6,33.8 ]

layer1 = tf.keras.layers.Dense(units = 1,input_shape=[1])
model = tf.keras.Sequential([layer1])

model.compile(loss ='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.005))
history = model.fit(celsius_q,fahrenheit_a, epochs= 2000)
print(model.predict([100]))
print("These are the layer variables: {}".format(layer1.get_weights()))
import matplotlib.pyplot as plt

plt.xlabel('Epoch Number')

plt.ylabel("Loss Magnitude")

plt.plot(history.history['loss'])