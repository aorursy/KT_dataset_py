import numpy as np

from numpy import random

from numpy import vstack

import pandas as pd

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

from PIL import Image

from numpy import expand_dims
from keras import backend

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Reshape

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import Conv2DTranspose

from keras.layers import LeakyReLU

from keras.layers import Dropout

from keras.optimizers import Adam

from keras.optimizers import RMSprop

from keras.layers import BatchNormalization

from keras.initializers import RandomNormal

from keras.constraints import Constraint

from keras.utils.vis_utils import plot_model

from keras.models import load_model
train_df = pd.read_csv('../input/train.csv')

train_df_7 = train_df[train_df['label']==7]

X = train_df_7.drop(['label'], axis=1).values

X = (X - 127.5) / 127.5 # scale from [0,255] to [-1,1]

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
class ClipConstraint(Constraint):

    """

    clip model weights to a given hypercube

    """

    def __init__(self, clip_value):

        self.clip_value = clip_value

    def __call__(self, weights):

        return backend.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):

        return {'clip_value': self.clip_value}
def wasserstein_loss(y_true, y_pred):

    """

    Wasserstein loss function

    """

    return backend.mean(y_true * y_pred)
def define_critic(in_shape=(n_x,n_x,n_c)):

    """

    Define the conv net for the critic

    """

    init = RandomNormal(stddev=0.02)    # weight initialization

    const = ClipConstraint(0.01)    # weight constraint

    model = Sequential()

    model.add(Conv2D(64,kernel_size=4,strides=2,padding='same',kernel_initializer=init,kernel_constraint=const,input_shape=in_shape))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(64,kernel_size=4,strides=2,padding='same',kernel_initializer=init,kernel_constraint=const))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())

    model.add(Dense(1,activation='linear'))

    opt = RMSprop(lr=0.00005)    # define optimizer

    model.compile(loss=wasserstein_loss,optimizer=opt)

    return model
critic = define_critic()

critic.summary()
def generate_real_samples(data, n_samples):

    """

    Pick 'n_samples' randomly from 'data'

    """

    idx = random.randint(low=0,high=data.shape[0],size=n_samples)

    X = data[idx]

    Y = -np.ones((n_samples,1))

    return X, Y
def define_generator(latent_dim):

    """

    Define the conv net for the generator

    """

    init = RandomNormal(stddev=0.02)    # weight initialization

    n_nodes = 128*7*7

    model = Sequential()

    model.add(Dense(n_nodes,kernel_initializer=init,input_dim=latent_dim)) # foundation for 7*7 image

    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((7,7,128)))

    model.add(Conv2DTranspose(128,kernel_size=4,strides=2,padding='same')) # up-sample to 14*14 image

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128,kernel_size=4,strides=2,padding='same',kernel_initializer=init)) # up-sample to 28*28 image

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1,kernel_size=7,activation='tanh',padding='same',kernel_initializer=init))

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

    Y = np.ones((n_samples,1))     # create class labels '1' for fake sample

    return X, Y
def define_gan(g_model, c_model):

    """

    This takes as arguments the generator and critic and creates the GAN subsuming these two models. 

    The weights in the critic are marked as not trainable, 

    which only affects the weights as seen by the GAN and not the standalone discriminator model.

    """

    c_model.trainable = False     # make weights in the critic not trainable

    model = Sequential()

    model.add(g_model)

    model.add(c_model)

    opt = RMSprop(lr=0.00005)

    model.compile(loss=wasserstein_loss, optimizer=opt)

    return model
gan = define_gan(generator, critic)

gan.summary()
def save_plot(examples, step, n=10):

    """

    This creates and save a plot of generated images

    """

    for i in range(n * n):

        plt.subplot(n, n, 1 + i)

        plt.axis('off')

        plt.imshow(examples[i, :, :, 0], cmap=plt.cm.binary)

    filename = 'generated_plot_e%04d.png' % (step+1)

    plt.savefig(filename)

    plt.close()
def summarize_performance(step, g_model, latent_dim, n_samples=100):

    """

    This plots generated images, save generator model

    """

    X_fake, Y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

    X_fake = (X_fake+1)/2.0

    save_plot(X_fake, step)

    filename = 'generator_model_%04d.h5' % (step+1)

    g_model.save(filename)
def train(g_model, c_model, gan_model, data, latent_dim, n_epochs=100, batch_size=64, n_critic=5):

    """

    This trains the combined generator and critic models in the GAN

    """

    batch_per_epoch = data.shape[0] // batch_size

    n_steps = batch_per_epoch * n_epochs

    half_batch = batch_size // 2

    for i in range(n_steps):

        for j in range(n_critic):

            X_real, Y_real = generate_real_samples(data, half_batch)   # randomly select real samples

            c_loss1 = c_model.train_on_batch(X_real, Y_real)

            X_fake, Y_fake = generate_fake_samples(g_model, latent_dim, half_batch)   # generate fake samples

            c_loss2 = c_model.train_on_batch(X_fake, Y_fake)

        X_gan = generate_latent_points(latent_dim, batch_size)   # as input for generator

        Y_gan = -np.ones((batch_size, 1))

        g_loss = gan_model.train_on_batch(X_gan, Y_gan)   # update generator via the discriminator's error

        print('>%d, c1=%.3f, c2=%.3f, g=%.3f' % (i+1, c_loss1, c_loss2, g_loss)) # summarize loss for batch        

        # evaluate the model performance, after some epochs

        if (i+1) % batch_per_epoch == 0: 

            summarize_performance(i, g_model, latent_dim)
latent_dim = 50

c_model = define_critic()

g_model = define_generator(latent_dim)

gan_model = define_gan(g_model, c_model)

data = X

train(g_model, c_model, gan_model, data, latent_dim)
img = Image.open('generated_plot_e6800.png')

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
latent_points = generate_latent_points(50, 64)

new_digits = g_model.predict(latent_points)

show_plot(new_digits, 8)