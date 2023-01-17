import math

from keras.callbacks import LambdaCallback

import keras.backend as K





class LRFinder:

    """

    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.

    See for details:

    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0

    """

    def __init__(self, model):

        self.model = model

        self.losses = []

        self.lrs = []

        self.best_loss = 1e9



    def on_batch_end(self, batch, logs):

        # Log the learning rate

        lr = K.get_value(self.model.optimizer.lr)

        self.lrs.append(lr)



        # Log the loss

        loss = logs['loss']

        self.losses.append(loss)



        # Check whether the loss got too large or NaN

        if math.isnan(loss) or loss > self.best_loss * 4:

            self.model.stop_training = True

            return



        if loss < self.best_loss:

            self.best_loss = loss



        # Increase the learning rate for the next batch

        lr *= self.lr_mult

        K.set_value(self.model.optimizer.lr, lr)



    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):

        num_batches = epochs * x_train.shape[0] / batch_size

        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))



        # Save weights into a file

        self.model.save_weights('tmp.h5')



        # Remember the original learning rate

        original_lr = K.get_value(self.model.optimizer.lr)



        # Set the initial learning rate

        K.set_value(self.model.optimizer.lr, start_lr)



        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))



        self.model.fit(x_train, y_train,

                        batch_size=batch_size, epochs=epochs,

                        callbacks=[callback])



        # Restore the weights to the state before model fitting

        self.model.load_weights('tmp.h5')



        # Restore the original learning rate

        K.set_value(self.model.optimizer.lr, original_lr)



    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):

        """

        Plots the loss.

        Parameters:

            n_skip_beginning - number of batches to skip on the left.

            n_skip_end - number of batches to skip on the right.

        """

        plt.ylabel("loss")

        plt.xlabel("learning rate (log scale)")

        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])

        plt.xscale('log')



    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):

        """

        Plots rate of change of the loss function.

        Parameters:

            sma - number of batches for simple moving average to smooth out the curve.

            n_skip_beginning - number of batches to skip on the left.

            n_skip_end - number of batches to skip on the right.

            y_lim - limits for the y axis.

        """

        assert sma >= 1

        derivatives = [0] * sma

        for i in range(sma, len(self.lrs)):

            derivative = (self.losses[i] - self.losses[i - sma]) / sma

            derivatives.append(derivative)



        plt.ylabel("rate of loss change")

        plt.xlabel("learning rate (log scale)")

        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])

        plt.xscale('log')

        plt.ylim(y_lim)
import pandas as pd

import numpy as np



def read_vectors(filename):

    return np.fromfile(filename, dtype=np.uint8).reshape(-1,401)



snk = np.vstack(tuple(read_vectors("../input/snake-eyes/snakeeyes_{:02d}.dat".format(nn))

                      for nn in range(2)))

snk_y = snk[:,0]

snk_X = snk[:,1:]



import matplotlib.pyplot as plt

plt.title(f'class == {snk_y[0]}')

plt.imshow(snk_X[0].reshape(20,20), cmap='Greys_r')
import keras

from keras.layers import *

from keras.models import Sequential



input_shape = (20, 20, 1)

batch_size = 128

num_classes = 12

epochs = 5

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.SGD(),

              metrics=['accuracy'])
lr_finder = LRFinder(model)

lr_finder.find(snk_X.reshape((200000, 20, 20, 1)),

               keras.utils.np_utils.to_categorical(snk_y)[:, 1:], 

               start_lr=10e-5, end_lr=1, batch_size=500, epochs=1)

lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)