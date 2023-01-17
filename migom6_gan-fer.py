!ls ../input/
import pandas as pd

df = pd.read_csv('../input/fer2013/fer2013.csv')
import keras

from keras import backend as K

from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam

from keras.layers import Activation, Dense, Input, Dropout

from keras.layers import Conv2D, Flatten

from keras.layers import Reshape, Conv2DTranspose

from keras.layers import BatchNormalization

from keras.layers.merge import concatenate

from keras.models import Model, Sequential

from keras.utils import to_categorical, plot_model

from keras.losses import binary_crossentropy

from keras.activations import relu



import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from PIL import Image
# predefined variables



image_height = 48

image_width = 48

n_classes = 7 # 7 emotion types

n_channels = 1 # grayscale image
# we don't have to worry that the image have to be from training/val/test, as we need to generate

# the images only

pixels = df['pixels']

emotions = df['emotion']
# extracting the dataset

real_image_dataset = []

real_image_emotions = []



for i in range(pixels.shape[0]) :

    if(i % 5000 == 0) :

        print(i)

    im = np.array(list(map(float, pixels[i].split(' '))))

    im = im.astype('float32') / 255.0

    im = np.reshape(im, (image_height, image_width, 1))

    real_image_dataset.append(im)



real_image_dataset = np.array(real_image_dataset)



real_image_emotions = emotions.values

real_image_emotions = to_categorical(real_image_emotions)
# just checking the dataset

fig, ax = plt.subplots(nrows=2, ncols=5)

iter_real_dataset_show = 0

for row in ax:

    for col in row:

        col.imshow(real_image_dataset[iter_real_dataset_show].reshape(image_height, image_width), cmap='gray')

        iter_real_dataset_show += 1

plt.show()
# discriminator model

def generate_discriminator(in_shape=(image_height, image_width, 1)) :

    model = Sequential()

    

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))



    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dropout(0.25))

    model.add(Dense(1, activation='sigmoid'))

    opt = Adam()

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



    return model
# generator model

def generate_generator(latent_dim) :

    model = Sequential()



    n_nodes = 128 * 3 * 3

    model.add(Dense(n_nodes, input_dim=latent_dim))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((3, 3, 128)))



    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))



    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))



    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))



    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))



    model.add(Conv2D(1, (7, 7), activation='tanh', padding='same'))

    

    return model
# GAN model

def generate_GAN(generator, discriminator): 

    discriminator.trainable = False

    model = Sequential()

    model.add(generator)

    model.add(discriminator)

    opt = Adam()

    model.compile(loss='binary_crossentropy', optimizer=opt)



    return model
# change images to be in the range from [-1, 1] 

# to match the output of generator



real_image_dataset = (real_image_dataset - 0.5) / 0.5
# select real samples

def generate_real_samples(dataset, n_samples): 

    idx = np.random.randint(0, dataset.shape[0], n_samples)

    X = dataset[idx]

    y = np.full(n_samples, 0.85)



    return X, y
# generate fake samples

def generate_fake_samples(generator, latent_dim, n_samples):

    x_input = generate_latent_points(latent_dim, n_samples)

    X = generator.predict(x_input)

    y = np.full(n_samples, 0.1)



    return X, y
# generate points in latent space as input for generator

def generate_latent_points(latent_dim, n_samples) :

    x_input = np.random.rand(latent_dim * n_samples)

    x_input = x_input.reshape(n_samples, latent_dim)



    return x_input
# training

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):

    batches_per_epoch = int(dataset.shape[0] / n_batch)

    half_batch = int(n_batch / 2)



    for i in range(n_epochs):

        for j in range(batches_per_epoch) :

            X_real, y_real = generate_real_samples(dataset, half_batch)

            discriminator.trainable = True

            d_loss1, _ = d_model.train_on_batch(X_real, y_real)

            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            

            X_gan = generate_latent_points(latent_dim, n_batch)

            y_gan = np.ones((n_batch, 1))

            discriminator.trainable = False

            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, batches_per_epoch, d_loss1, d_loss2, g_loss))
latent_dim = 100

discriminator = generate_discriminator()

generator = generate_generator(latent_dim)

gan_model = generate_GAN(generator, discriminator)

train(generator, discriminator, gan_model, real_image_dataset, latent_dim)
from matplotlib import pyplot



def show_plot(examples, n) :

    for i in range(n * n) :

        pyplot.subplot(n, n, i + 1)

        pyplot.axis('off')

        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')

    pyplot.show()

    

latent_points = generate_latent_points(100, 100)

X = generator.predict(latent_points)

show_plot(X, 7)