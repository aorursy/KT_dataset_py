# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sys

sys.path.append('../input')

import tensorflow as tf

import datasets

from sklearn.preprocessing import scale, minmax_scale



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import random

def generate_x_y_data_v3(isTrain, batch_size):

    """

    Similar to the "v2" function, but here we generate a signal

    with noise in the X values. Plus,

    the lenght of the examples is of 30 rather than 10.

    So we have 60 total values for past and future.

    """

    seq_length = 1000

    x, y = datasets.generate_x_y_data_two_freqs(

        isTrain, batch_size, seq_length=seq_length)

    noise_amount = random.random() * 0.15 + 0.10

    x = x + noise_amount * np.random.randn(seq_length, batch_size, 1)



    avg = np.average(x)

    std = np.std(x) + 0.0001

    x = x - avg

    y = y - avg

    x = x / std / 2.5

    y = y / std / 2.5



    return x, y
x, y = generate_x_y_data_v3(isTrain=True, batch_size=1)
x.shape, y.shape
def plot_signals(x,y,limit=100):

    plt.plot(np.arange(len(x))[:limit], x.ravel().ravel()[:limit], label='x')

    plt.plot(np.arange(len(y))[:limit], y.ravel().ravel()[:limit], label='y')

    plt.legend()

    plt.show()
plot_signals(x,y)
class DenoiseAE(tf.keras.models.Model):

    

    def __init__(self, nvars):

        super(DenoiseAE, self).__init__(name='Denoise Recurrent AE')

        self.encoder = tf.keras.layers.Conv1D(50, 4, input_shape=(4,1))

        self.encoder2 = tf.keras.layers.GRU(40)

        self.encoder3 = tf.keras.layers.LSTM(20, return_sequences=True)

        self.decoder = tf.keras.layers.GRU(10, return_sequences=True)

        self.decoder2 = tf.keras.layers.Dense(nvars)

        self.nvars = nvars

        

    def call(self, inputs):

        output = self.encoder(inputs)

        output = self.encoder2(output)

#         output3 = self.encoder3(inputs)

#         output = tf.keras.layers.Concatenate()([output1, output2, output3])

#         output = self.decoder(output)

        output = self.decoder2(output)

        output = tf.keras.layers.Reshape((self.nvars, 1))(output)

        return output
tf.keras.backend.clear_session()
nvars = 5

model = DenoiseAE(nvars)



model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.mse)



x = x.reshape((-1,nvars,1))

y = y.reshape((-1,nvars,1))

model.fit(x, y, batch_size=32, epochs=500, verbose=0)
yhat = model.predict(x)
x,y,yhat = [el.reshape((-1, 1)) for el in [x,y,yhat]]
plot_signals(x,y, limit=50)

plot_signals(yhat, y, limit=50)