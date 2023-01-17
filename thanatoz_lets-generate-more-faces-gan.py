import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))

from PIL import Image



import keras

from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model

from keras.optimizers import RMSprop

from keras import backend as K
def load_img(PATH): 

    return np.array(Image.open(PATH).resize((64,64)))
img_base = '../input/test/'

images = os.listdir(img_base)
test_images = np.array([load_img(os.path.join(img_base, i)) for i in images])
# Defining hyper-parameters



img_rows = 64

img_cols = 64

channels = 3

img_shape = (img_rows, img_cols, channels)

latent_dim = 100

n_critic = 5

clip_value = 0.01

optimizer = RMSprop(lr=0.00005)
def build_critic():



    model = Sequential()



    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.75))

    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))

    model.add(ZeroPadding2D(padding=((0,1),(0,1))))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.75))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.75))

    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.75))

    model.add(Flatten())

    model.add(Dense(1))



    #         model.summary()



    img = Input(shape=img_shape)

    validity = model(img)



    return Model(img, validity)



def build_generator():



    model = Sequential()



    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dim))

    model.add(Reshape((8, 8, 128)))

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=4, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=4, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    model.add(UpSampling2D())

    model.add(Conv2D(64, kernel_size=4, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    model.add(Conv2D(channels, kernel_size=4, padding="same"))

    model.add(Activation("tanh"))



    #         model.summary()



    noise = Input(shape=(latent_dim,))

    img = model(noise)



    return Model(noise, img)



def wasserstein_loss(y_true, y_pred):

        return K.mean(y_true * y_pred)

    

    

def sample_images(epoch=None):

        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, latent_dim))

        gen_imgs = generator.predict(noise)



        # Rescale images 0 - 1

        gen_imgs = 0.5 * gen_imgs + 0.5



        fig, axs = plt.subplots(r, c)

        cnt = 0

        for i in range(r):

            for j in range(c):

                axs[i,j].imshow(gen_imgs[cnt, :,:])

                axs[i,j].axis('off')

                cnt += 1

        fig.savefig("./faces_%d.png" % epoch)

        plt.close()
critic = build_critic()

critic.compile(loss=wasserstein_loss,

                optimizer=optimizer,

                metrics=['accuracy'])

critic.trainable = False



generator = build_generator()

z = Input(shape=(latent_dim,))

img = generator(z)

valid = critic(img)

combined = Model(z, valid)

combined.compile(loss=wasserstein_loss,

                    optimizer=optimizer,

                    metrics=['accuracy'])
def train(epochs, batch_size=128, sample_interval=50):

    

    # Load the dataset

    X_train = test_images

    # Rescale -1 to 1

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5



    # Adversarial ground truths

    valid = -np.ones((batch_size, 1))

    fake = np.ones((batch_size, 1))



    for epoch in range(epochs):



        for _ in range(n_critic):



            # ---------------------

            #  Train Discriminator

            # ---------------------



            # Select a random batch of images

            idx = np.random.randint(0, X_train.shape[0], batch_size)

    #                 print(idx)

            imgs = X_train[idx]



            # Sample noise as generator input

            noise = np.random.normal(0, 1, (batch_size, latent_dim))



            # Generate a batch of new images

            gen_imgs = generator.predict(noise)



            # Train the critic

            d_loss_real = critic.train_on_batch(imgs, valid)

            d_loss_fake = critic.train_on_batch(gen_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)



            # Clip critic weights

            for l in critic.layers:

                weights = l.get_weights()

                weights = [np.clip(w, -clip_value, clip_value) for w in weights]

                l.set_weights(weights)



        # ---------------------

        #  Train Generator

        # ---------------------



        g_loss = combined.train_on_batch(noise, valid)



        # Plot the progress

        if epoch%1000==0:

            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            generator.save('./generator_%d.h5'%(epoch))

            generator.save_weights('./generator_weights_%d.h5'%(epoch))

            critic.save('./critic_%d.h5'%(epoch))

            critic.save_weights('./critic_weights_%d.h5'%(epoch))

        # If at save interval => save generated image samples

#         if epoch % sample_interval == 0:

# #             generator.save('./generator_%d.h5'%(epoch))

#             generator.save_weights('./generator_weights_%d.h5'%(epoch))

            sample_images(epoch)

    generator.save('./generator_%d.h5'%(epoch))

    generator.save_weights('./generator_weights_%d.h5'%(epoch))

    critic.save('./critic_%d.h5'%(epoch))

    critic.save_weights('./critic_weights_%d.h5'%(epoch))
train(epochs=50000, batch_size=128, sample_interval=1000)
r, c = 5, 5

noise = np.random.normal(0, 1, (r * c, latent_dim))

gen_imgs = generator.predict(noise)



# Rescale images 0 - 1

gen_imgs = 0.5 * gen_imgs + 0.5
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

for i,ax in enumerate(axes.flat): ax.imshow(gen_imgs[i])
r, c = 5, 5

noise = np.random.normal(0, 1, (r * c, latent_dim))

gen_imgs = generator.predict(noise)



# Rescale images 0 - 1

gen_imgs = 0.5 * gen_imgs + 0.5



fig, axes = plt.subplots(2, 2, figsize=(6, 6))

for i,ax in enumerate(axes.flat): ax.imshow(gen_imgs[i])