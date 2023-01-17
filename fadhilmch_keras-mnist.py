# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from IPython.display import clear_output



# Any results you write to the current directory are saved as output.
from keras.layers import Input, Dense

from keras.models import Model

from keras.callbacks import TensorBoard

import keras



encoding_dim = 50



# this is our input placeholder

input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input

encoded = Dense(encoding_dim, activation='sigmoid')(input_img)

# "decoded" is the lossy reconstruction of the input

decoded = Dense(784, activation='sigmoid')(encoded)



# this model maps an input to its reconstruction

autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation

encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input

encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model

decoder_layer = autoencoder.layers[-1]

# create the decoder model

decoder = Model(encoded_input, decoder_layer(encoded_input))
ada = keras.optimizers.Adadelta(lr=10, rho=0.95, epsilon=None, decay=0.0)

autoencoder.compile(optimizer=ada, loss='binary_crossentropy')
x_train = np.genfromtxt('../input/bindigit_trn.csv', delimiter=',')

x_test = np.genfromtxt('../input/bindigit_tst.csv', delimiter=',')
class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.i += 1

        

        clear_output(wait=True)

        plt.plot(self.x, self.losses, label="loss")

        plt.plot(self.x, self.val_losses, label="val_loss")

        plt.legend()

        plt.show();

        

plot_losses = PlotLosses()
autoencoder.fit(x_train, x_train,

                epochs=1000,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test, x_test),

                callbacks=[plot_losses])
# encode and decode some digits

# note that we take them from the *test* set

encoded_imgs = encoder.predict(x_test)

decoded_imgs = decoder.predict(encoded_imgs)
# use Matplotlib (don't ask)

import matplotlib.pyplot as plt



n = 10  # how many digits we will display

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()