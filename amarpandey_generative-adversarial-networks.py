import gzip

import pickle

import sys, os

import numpy as np

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0 * 2 - 1, x_test / 255.0 * 2 - 1
N, W, H = x_train.shape

D = W * H

x_test = x_test.reshape(-1, D)

x_train = x_train.reshape(-1, D)



assert(len(x_test) == len(y_test))

assert(len(x_train) == len(y_train))



print("No of training examples: ", len(x_train))

print("shape of training examples: ", x_train.shape)
# not sure about this

latent_dim = 100
from tensorflow.keras import Input

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout

from tensorflow.keras.optimizers import Adam
# Get the generator model

def build_generator(latent_dim):

    i = Input(shape=(latent_dim,))

    x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)

    x = BatchNormalization(momentum=0.7)(x)

    x = Dropout(0.3)(x)

    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)

    x = BatchNormalization(momentum=0.7)(x)

    x = Dropout(0.3)(x)

    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)

    x = BatchNormalization(momentum=0.7)(x)

    x = Dense(D, activation='tanh')(x)

    

    model = Model(i, x)

    

    return model
def build_discriminator(img_size):

    i = Input(shape=(img_size,))

    x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)

    x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)

    x = Dense(1, activation='sigmoid')(x)

    

    model = Model(i, x)

    

    return model
# Build and compile the discriminator

d_model = build_discriminator(D)

d_model.compile(

    loss='binary_crossentropy',

    optimizer=Adam(0.0001, 0.5),

    metrics=['accuracy'])



# Build and compile the combined model

g_model = build_generator(latent_dim)



# Create an input to represent noise sample from latent space

z = Input(shape=(latent_dim,))



# Pass noise through generator to get an image

img = g_model(z)



# Make sure only the generator is trained

d_model.trainable = False



# The true output is fake, but we label them real!

fake_pred = d_model(img)



# Create the combined model object

combined_model = Model(z, fake_pred)



# Compile the combined model

combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))
d_losses = []

g_losses = []



epochs = 30000

batch_size = 16

sample_period = 200



ones = np.random.uniform(low=0.7, high=1.1, size=batch_size)

zeros = np.random.uniform(low=0.0, high=0.3, size=batch_size)



if not os.path.exists('gan_images'):

    os.makedirs('gan_images')
# record images after sample period

def sample_images(epoch):

    

    rows, cols = 5, 5

    noise = np.random.randn(rows * cols, latent_dim)

    

    assert(noise.shape == (25, latent_dim))

    

    imgs = g_model.predict(noise)



    # Rescale images 0 - 1

    imgs = 0.5 * imgs + 0.5



    fig, axs = plt.subplots(rows, cols)

    idx = 0

    for i in range(rows):

        for j in range(cols):

            axs[i,j].imshow(imgs[idx].reshape(H, W), cmap='gray')

            axs[i,j].axis('off')

            idx += 1

    fig.savefig("gan_images/%d.png" % epoch, dpi=400)

    plt.close()
for epoch in range(epochs):

    

    ###########################

    ### Train discriminator ###

    ###########################

    

    # Select a random batch of images

    idx = np.random.randint(0, x_train.shape[0], batch_size)

    real_imgs = x_train[idx]

    

    # Generate fake images

    noise = np.random.randn(batch_size, latent_dim)

    fake_imgs = g_model.predict(noise)    



    # Train the discriminator

    # both loss and accuracy are returned

    d_loss_real, d_acc_real = d_model.train_on_batch(real_imgs, ones)

    d_loss_fake, d_acc_fake = d_model.train_on_batch(fake_imgs, zeros)

    

    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    # d_acc  = 0.5 * (d_acc_real + d_acc_fake)

    

    ###########################

    ##### Train generator #####

    ###########################



    noise = np.random.randn(batch_size, latent_dim)

    g_loss = combined_model.train_on_batch(noise, ones)

    

    # do it again!

    noise = np.random.randn(batch_size, latent_dim)

    g_loss = combined_model.train_on_batch(noise, ones)



    # Save the losses

    d_losses.append(d_loss)

    g_losses.append(g_loss)

    

    if epoch % 100 == 0:

        print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f},  g_loss: {g_loss:.2f}")

        

    if epoch % sample_period == 0:

        sample_images(epoch)
sample_images(epoch+1)
plt.plot(g_losses, label='g_losses')

plt.plot(d_losses, label='d_losses')

plt.legend()
from skimage.io import imread

a = imread('gan_images/0.png')

plt.imshow(a)
a = imread('gan_images/2000.png')

plt.imshow(a)
a = imread('gan_images/4000.png')

plt.imshow(a)
a = imread('gan_images/18000.png')

plt.imshow(a)
a = imread('gan_images/20000.png')

plt.imshow(a)
a = imread('gan_images/28000.png')

plt.imshow(a)
a = imread('gan_images/30000.png')

plt.imshow(a)