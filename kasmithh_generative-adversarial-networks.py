# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense



def discriminator_model(n_inputs = 2):

    model = Sequential()

    model.add(Dense(10, activation = 'relu', input_dim = n_inputs))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

discriminator = discriminator_model()
def define_generator(latent_dim, n_outputs = 2):

    model = Sequential()

    model.add(Dense(10, activation = 'relu',kernel_initializer='he_uniform', input_dim=latent_dim))

    model.add(Dense(n_outputs, activation = 'linear'))

    return model



latent_dim = 5

generator = define_generator(latent_dim)
def define_gan(generator, discriminator):

    discriminator.trainable = False

    model = Sequential()

    model.add(generator)

    model.add(discriminator)

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

    return model



gan = define_gan(generator, discriminator)
from numpy import zeros

from numpy import ones

from numpy import hstack

from numpy.random import rand

from numpy.random import randn



def generate_real(n):

    X1 = rand(n) - 0.5

    X2 = X1 * X1

    X2 = X2.reshape(n, 1)

    X1 = X1.reshape(n, 1)

    X = hstack((X1, X2))

    y = ones(n)

    return X, y 



def generate_fake(n):

    X = -1 + rand(n)

    X = X.reshape(n, 1)

    y = zeros(n)

    return X, y
def generate_latent_points(latent_dim, n):

    x_input = randn(latent_dim * n)

    x_input = x_input.reshape(n, latent_dim)

    return x_input



def generate_fake_samples(generator, latent_dim, n):

    x_input = generate_latent_points(latent_dim, n)

    X = generator.predict(x_input)

    y = zeros(n)

    return X, y
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):

    # prepare real samples

    x_real, y_real = generate_real(n)

    # evaluate discriminator on real examples

    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)

    # prepare fake examples

    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)

    # evaluate discriminator on fake examples

    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

    # summarize discriminator performance

    print(epoch, acc_real, acc_fake)

    # scatter plot real and fake data points

    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')

    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')

    pyplot.show()
def train(g_model, d_model, gan_model, latent_dim, n_epochs = 1000, n_batch = 100):

    half_batch = int(n_batch/2)

    for i in range(n_epochs):

        x_real, y_real = generate_real(half_batch)

        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

        d_model.train_on_batch(x_real, y_real)

        d_model.train_on_batch(x_fake, y_fake)

        x_gan = generate_latent_points(latent_dim, n_batch)

        y_gan = ones(n_batch)

        gan_model.train_on_batch(x_gan, y_gan)
train(g_model = generator, d_model = discriminator, gan_model = gan,

      latent_dim = latent_dim, n_epochs = 1000, n_batch = 100)
summarize_performance(1000, generator, discriminator, latent_dim, n = 100)