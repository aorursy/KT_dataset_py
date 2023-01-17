import numpy as np

from numpy import random

from numpy import vstack

import pandas as pd

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

from PIL import Image
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Reshape

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import Conv2DTranspose

from keras.layers import LeakyReLU

from keras.layers import Dropout

from keras.optimizers import Adam

from keras.utils.vis_utils import plot_model

from keras.models import load_model
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

X_train = train_df.drop(['label'], axis=1).values

X_test = test_df.values

X = np.concatenate((X_train, X_test))

X = X / 255. # standardize the values in the datasets by dividing by 255

print(X.shape)
# reshape flattened data into tensors

n_x = 28     # size of image n_x by n_x

n_c = 1      # number of channels

X = X.reshape((-1, n_x, n_x, n_c))

print(X.shape)
# pick some random digits from the dataset X and look at them

plt.figure(figsize=(10,6))

n_digits = 60

select = random.randint(low=0,high=X.shape[0],size=n_digits)

for i, index in enumerate(select):  

    plt.subplot(5, 12, i+1)

    plt.imshow(X[index].reshape((28, 28)), cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
def define_discriminator(in_shape=(n_x,n_x,n_c)):

    """

    Define the conv net for the discriminator

    """

    model = Sequential()

    model.add(Conv2D(64,kernel_size=3,strides=2,padding='same',input_shape=in_shape))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(rate=0.4))

    model.add(Conv2D(64,kernel_size=3,strides=2,padding='same'))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(rate=0.4))

    model.add(Flatten())

    model.add(Dense(1,activation='sigmoid'))

    opt = Adam(lr=0.0002,beta_1=0.5) # define optimizer

    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

    return model
discriminator = define_discriminator()

discriminator.summary()
def generate_real_samples(data, n_samples):

    """

    Pick 'n_samples' randomly from 'data'

    """

    idx = random.randint(low=0,high=data.shape[0],size=n_samples)

    X = data[idx]

    Y = np.ones((n_samples,1))

    return X, Y
def define_generator(latent_dim):

    """

    Define the conv net for the generator

    """

    n_nodes = 128*7*7

    model = Sequential()

    model.add(Dense(n_nodes,input_dim=latent_dim)) # foundation for 7*7 image

    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((7,7,128)))

    model.add(Conv2DTranspose(128,kernel_size=4,strides=2,padding='same')) # up-sample to 14*14 image

    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128,kernel_size=4,strides=2,padding='same')) # up-sample to 28*28 image

    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1,kernel_size=7,activation='sigmoid',padding='same'))

    return model
latent_dim = 100 # define size of latent space

generator = define_generator(latent_dim)

generator.summary()
def generate_latent_points(latent_dim, n_samples):

    """

    This generates points in the latent space as input for the generator

    """

    x_input = random.randn(latent_dim * n_samples)

    x_input = x_input.reshape(n_samples, latent_dim)     # reshape into a batch of inputs for the network

    return x_input
def generate_fake_samples(g_model, latent_dim, n_samples):

    """

    Generate 'n_samples' of fake samples from the generator

    """

    X_input = generate_latent_points(latent_dim, n_samples)

    X = g_model.predict(X_input)    # generator predicts output

    Y = np.zeros((n_samples,1))     # create class labels '0' for fake sample

    return X, Y
def define_gan(g_model, d_model):

    """

    This takes as arguments the generator and discriminator and creates the GAN subsuming these two models. 

    The weights in the discriminator are marked as not trainable, 

    which only affects the weights as seen by the GAN and not the standalone discriminator model.

    """

    d_model.trainable = False     # make weights in the discriminator not trainable

    model = Sequential()

    model.add(g_model)

    model.add(d_model)

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model
gan = define_gan(generator, discriminator)

gan.summary()
def save_plot(examples, epoch, n=10):

    """

    This creates and save a plot of generated images

    """

    for i in range(n * n):

        plt.subplot(n, n, 1 + i)

        plt.axis('off')

        plt.imshow(examples[i, :, :, 0], cmap=plt.cm.binary)

    filename = 'generated_plot_e%03d.png' % (epoch+1)

    plt.savefig(filename)

    plt.close()
def summarize_performance(epoch, g_model, d_model, data, latent_dim, n_samples=100):

    """

    This evaluates the discriminator, plot generated images, save generator model

    """

    X_real, Y_real = generate_real_samples(data, n_samples)

    _, acc_real = d_model.evaluate(X_real, Y_real, verbose=0)   # evaluate discriminator on real samples

    X_fake, Y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

    _, acc_fake = d_model.evaluate(X_fake, Y_fake, verbose=0)   # evaluate discriminator on fake samples

    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))    # summarize discriminator performance

    save_plot(X_fake, epoch)

    filename = 'generator_model_%03d.h5' % (epoch + 1)

    g_model.save(filename)
def train(g_model, d_model, gan_model, data, latent_dim, n_epochs=100, batch_size=256):

    """

    This trains the combined generator and discriminator models in the GAN

    """

    batch_per_epoch = data.shape[0] // batch_size

    half_batch = batch_size // 2

    for i in range(n_epochs):

        for j in range(batch_per_epoch):

            X_real, Y_real = generate_real_samples(data, half_batch)   # randomly select real samples

            X_fake, Y_fake = generate_fake_samples(g_model, latent_dim, half_batch)   # generate fake samples

            X, Y = vstack((X_real, X_fake)), vstack((Y_real, Y_fake))  # create training set for discriminator

            d_loss, _ = d_model.train_on_batch(X, Y)   # update discriminator model weights

            X_gan = generate_latent_points(latent_dim, batch_size)   # as input for generator

            Y_gan = np.ones((batch_size, 1))

            g_loss = gan_model.train_on_batch(X_gan, Y_gan)   # update generator via the discriminator's error

            # print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_per_epoch, d_loss, g_loss)) # summarize loss for batch

        # evaluate the model performance, after some epochs

        if (i+1) % 10 == 0: 

            summarize_performance(i, g_model, d_model, data, latent_dim)
latent_dim = 100

d_model = define_discriminator()

g_model = define_generator(latent_dim)

gan_model = define_gan(g_model, d_model)

data = X

train(g_model, d_model, gan_model, data, latent_dim)
img = Image.open('generated_plot_e100.png')

plt.figure(figsize = (10,10))

plt.imshow(img)
def show_plot(examples, n):

    """

    This shows the plots from the GAN

    """

    plt.figure(figsize=(10,10))

    for i in range(n * n):

        plt.subplot(n, n, 1 + i)

        plt.axis('off')

        plt.imshow(examples[i, :, :, 0], cmap=plt.cm.binary)

    plt.show()
latent_points = generate_latent_points(100, 64)

new_digits = g_model.predict(latent_points)

show_plot(new_digits, 8)