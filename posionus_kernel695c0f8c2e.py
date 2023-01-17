# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image
from tensorflow import keras

from matplotlib import image
from matplotlib import pyplot as plt
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sizes = pd.DataFrame(columns=['x', 'y'])
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5000]:
        img = Image.open(os.path.join(dirname, filename))
        sizes.loc[filename, ('x','y')] = img.size

sizes['x'].hist(bins=100)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:10]:
        im = Image.open(os.path.join(dirname, filename))
        im = im.resize((128, 64))
        im = np.asarray(im)
        plt.imshow(im)
        plt.show()


raw_images_list = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        im = Image.open(os.path.join(dirname, filename))
        im = im.resize((64, 64))
        im = np.asarray(im)
        raw_images_list.append(im)

images = np.array(np.stack(raw_images_list) / 255, dtype='float32')
images.shape
images.dtype
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
def define_discriminator(in_shape=(64,64,3)):
    model = Sequential()
    model.add(Conv2D(64, (4,4), strides=(2, 2), padding='same', input_shape=in_shape))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.4))
    model.add(Conv2D(128, (4,4), strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.4))
    model.add(Conv2D(256, (4,4), strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.4))
    model.add(Conv2D(512, (4,4), strides=(1, 1), padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0003, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = 1024 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 1024)))
    
    model.add(Conv2DTranspose(512, (4,4), strides=(2,2), padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 14x14
    model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
#     model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2DTranspose(3, (4,4), activation='sigmoid', strides=(2,2), padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Conv2D(3, (7,7), activation='sigmoid', padding='same'))
    return model

def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0003, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y

def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=2):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :])
        plt.show()
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%%%, fake: %.0f%%%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epo = 128
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
        # evaluate the model performance, sometimes
        if (i+1) % 1 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


latent_dim = 64
# create the discriminator
d_model = define_discriminator()
print(d_model.summary())
# create the generator
g_model = define_generator(latent_dim)
print(g_model.summary())
# create the gan
gan_model = define_gan(g_model, d_model)

print(gan_model.summary())
# load image data
dataset = images
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128)
# size of the latent space
latent_dim = 64
# define the discriminator model
model = g_model
# generate samples
n_samples = 9
X, _ = generate_fake_samples(model, latent_dim, n_samples)
plt.figure(figsize=(8,8))
# plot the generated samples
for i in range(n_samples):
    
    # define subplot
    plt.subplot(3, 3, i+1)
    # turn off axis labels
    plt.axis('off')
    # plot single image
    plt.imshow(X[i, :, :])
# show the figure
plt.show()