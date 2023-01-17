# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install tensorflow-gpu==1.14.0



from random import random 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread

from skimage.transform import resize



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



dirname = [ i for i in os.walk('/kaggle/input')][1][0]

filenames = [ i for i in os.walk('/kaggle/input')][1][2]

filenames = [f for f in filenames if f.endswith('.png')]



img_rows = 128

img_cols = 128

channels = 3

img_shape = (img_rows, img_cols, channels)

latent_dim = 100

batch_size=64

X = np.zeros((len(filenames),img_rows,img_cols,3))

from tqdm import tqdm

bad_files = []

for num,f in tqdm(enumerate(filenames)):



    img = imread('/kaggle/input/pokemon/'+f)

    X[num] =resize(img,(img_rows,img_cols),anti_aliasing=True)[:,:,:3]

from os import listdir



import matplotlib.pyplot as plt

from PIL import Image

from PIL.Image import BICUBIC

from skimage import io

from tqdm import tqdm 



from keras.datasets import mnist



from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose

from keras.models import Sequential, Model

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

import time

import os

from PIL import Image



import tensorflow as tf

from scipy.misc import imread, imsave

import cv2

from matplotlib import pyplot as plt





import sys

import os

from PIL import Image

from glob import glob

import math

import numpy as np

from copy import copy


def build_generator():

    #4*4*512

    model = Sequential()

    model.add(Dense(latent_dim * 4 * 4, activation="relu", input_dim=latent_dim))

    model.add(Reshape((4, 4, latent_dim)))

    model.add(UpSampling2D())

    #8*8*256

    model.add(Conv2D(256, kernel_size=5, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    model.add(UpSampling2D())

    #16*16*128

    model.add(Conv2D(128, kernel_size=5, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    model.add(UpSampling2D())

    #32*32*64

    model.add(Conv2D(64, kernel_size=5, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    model.add(UpSampling2D())

    #64*64*32

    model.add(Conv2D(32, kernel_size=5, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    model.add(UpSampling2D())

    #128*128*3

    model.add(Conv2D(3, kernel_size=5, padding="same"))

    model.add(Activation("tanh"))

    model.summary()



    noise = Input(shape=(latent_dim,))

    img = model(noise)

    return Model(noise, img)

'''

def build_generator():

    model = Sequential()

    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dim))

    model.add(Reshape((8, 8, 128)))

    model.add(Conv2DTranspose(filters=128, kernel_size=(3, 3),

                                  strides=(2, 2), padding='same',

                                  data_format='channels_last',

                                  kernel_initializer='glorot_uniform'))

    model.add(BatchNormalization(momentum=0.5))

    model.add(LeakyReLU(alpha=.1))

    model.add(Conv2DTranspose(filters=128, kernel_size=(3, 3),

                                  strides=(2, 2), padding='same',

                                  data_format='channels_last',

                                  kernel_initializer='glorot_uniform'))

    model.add(BatchNormalization(momentum=0.5))

    model.add(LeakyReLU(alpha=.1))

    model.add(Conv2D(channels, kernel_size=3, padding="same"))

    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(latent_dim,))

    img = model(noise)

    return Model(noise, img)

'''

def build_discriminator():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dropout(0.5))

    model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))

    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))

    model.add(BatchNormalization(momentum=0.5))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))

    model.add(BatchNormalization(momentum=0.5))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.50))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))

    model.add(BatchNormalization(momentum=0.5))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))

    model.add(BatchNormalization(momentum=0.5))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.50))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)

    validity = model(img)

    return Model(img, validity)



def train_gain(learning_rate=.0002,epochs=1000,save_interval=1000,disc_prob = .5):

    def save_imgs(epoch):

        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, latent_dim))

        gen_imgs0 = generator.predict(noise)



        # Rescale images 0 - 1

        gen_imgs = gen_imgs0*.5 +1



        #fig, axs = plt.subplots(r, c)

        #cnt = 0

        #for i in range(r):

            #for j in range(c):

                #axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')

                #axs[i, j].axis('off')





                #cnt += 1

        #fig.savefig("images/mnist_%d.png" % epoch)



        #plt.close()

        plt.imshow(((gen_imgs0[1]+1)*127.5).astype('uint8'))

        plt.show()

        #plt.imshow(gen_imgs[0])



        plt.show()

    def train(epochs, batch_size=128, save_interval=50,disc_prob=.5):

        os.makedirs('images', exist_ok=True)

        E,D,G = [],[],[]

        # Load the dataset

        #(X_train, _), (_, _) = mnist.load_data()



        # Rescale -1 to 1

        X_train = copy(X)

        X_train = X_train*2 -1

        #X_train = np.expand_dims(X_train, axis=3)



        # Adversarial ground truths

        valid = np.ones((batch_size, 1))

        fake = np.zeros((batch_size, 1))

        D_loss =[1,1]

        loss_ratio = 1.0

        for epoch in range(epochs):

            # Select a random real images

            idx = np.random.randint(0, X_train.shape[0], batch_size)

            real_imgs = X_train[idx]



            # Sample noise and generate a batch of fake images

            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            fake_imgs = generator.predict(noise)



            # Train the discriminator

            frac = random()

            if frac < disc_prob:

                D_loss_real = discriminator.train_on_batch(real_imgs, valid)

                D_loss_fake = discriminator.train_on_batch(fake_imgs, fake)

                D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

            # Train the generator

            g_loss = combined.train_on_batch(noise, valid)



            loss_ratio = loss_ratio*3/4 + g_loss/D_loss[0]/4



            if loss_ratio > 10:

                disc_prob = disc_prob - .01

            elif loss_ratio <.01:

                disc_prob = disc_prob + .01

            # If at save interval

            if epoch % save_interval == 0:

                # Print the progress

                

                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] Disc Prob %f  loss ratio %f" % (epoch, D_loss[0], 100 * D_loss[1], g_loss,disc_prob,loss_ratio))

                # Save generated image samples

                save_imgs(epoch)

                E.append(epoch)

                D.append(D_loss[0])

                G.append(g_loss)

        return E,D,G

    optimizer = Adam(learning_rate, 0.5)



    # build discriminator

    discriminator = build_discriminator()

    discriminator.compile(loss='binary_crossentropy',

                          optimizer=optimizer,

                          metrics=['accuracy'])





    # build generator

    generator = build_generator()

    generator.compile(loss='binary_crossentropy',

                          optimizer=optimizer)

    z = Input(shape=(latent_dim,))

    img = generator(z)



    # For the combined model we will only train the generator

    discriminator.trainable = False



    # The discriminator takes generated images as input and determines validity

    valid = discriminator(img)



    # The combined model  (stacked generator and discriminator)

    # Trains the generator to fool the discriminator

    combined = Model(z, valid)

    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    start = time.time()



    E,D,G = train(epochs=epochs, batch_size=batch_size, save_interval=save_interval,disc_prob=disc_prob)



    end = time.time()

    elapsed_train_time = 'elapsed training time: {} min, {} sec '.format(int((end - start) / 60),

                                                                         int((end - start) % 60))

    print(elapsed_train_time)

    plt.scatter(E,G)

    plt.scatter(E,D)

    plt.show()
train_gain(learning_rate = .0002,epochs = 200000,save_interval=1000,disc_prob=.15)