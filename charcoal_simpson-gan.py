import numpy as np
import pandas as pd 
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

img_rows = 128
img_cols = 128
channels = 3
img_shape = (img_rows, img_cols, channels)
latent_dim = 100

optimizer = Adam(0.0002, 0.5)
baseDir = "../input/cropped/"

train_x = []

characterNames =[]
counter = 0
for imgname in  os.listdir(baseDir + '/'):
    if ".png" in imgname:
        img = Image.open(baseDir +'/' + imgname)
        img.thumbnail((img_shape[0],img_shape[1]), Image.ANTIALIAS)
        
        x = np.asarray(img)
        x = x / 255

        train_x.append(x)
        counter += 1
     
print("loaded!")
X_train = np.array(train_x)
print(X_train.shape)
generator = Sequential()

iDimX = int(img_rows / 4)
iDimY = int(img_cols / 4)

generator.add(Dense(128 * iDimX * iDimY, activation="relu", input_dim=latent_dim))
generator.add(Reshape((iDimX, iDimY, 128)))
generator.add(UpSampling2D())
for i in range(1):
    generator.add(Conv2D(128, kernel_size=3, padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation("relu"))
generator.add(UpSampling2D())
for i in range(1):
    generator.add(Conv2D(64, kernel_size=5, padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation("relu"))
for i in range(1):
    generator.add(Conv2D(channels, kernel_size=7, padding="same"))
generator.add(Activation("tanh"))

generator.summary()
discriminator = Sequential()

discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
for i in range(1):
    discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
for i in range(1):
    discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
for i in range(1):
    discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))

discriminator.summary()
discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
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
# Adversarial ground truths
batch_size = 100
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

r, c = 4, 4
noise = np.random.normal(0, 1, (r*c, latent_dim))

for epoch in range(20000):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random half of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    # Sample noise and generate a batch of new images
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = generator.predict(noise)

    # Train the discriminator (real classified as ones and generated as zeros)
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    # Train the generator (wants discriminator to mistake images as real)
    g_loss = combined.train_on_batch(noise, valid)

    # Plot the progress
    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    # If at save interval => save generated image samples
    if epoch % 2000 == 0:
        generator.save("g{}.h5".format(epoch))
        
        gen_imgs = generator.predict(noise)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        plt.show()
        plt.close()
generator.save("final.h5".format(epoch))
num_tests = 50
noise = np.random.normal(0, 1, (num_tests, latent_dim))
gen_imgs = generator.predict(noise)

c = 1
for img in gen_imgs:
    i = img * 255
    Image.fromarray(i.astype('uint8')).save("f_{}.png".format(c))
    c += 1
