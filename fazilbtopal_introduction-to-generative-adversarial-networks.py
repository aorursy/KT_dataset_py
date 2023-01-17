from glob import glob

from PIL import Image

from IPython import display

import tensorflow.keras as kr

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import imageio

import os
BATCH_SIZE = 128

LATENT_DIM = 100

SAMPLE_INTERVAL = 200

EPOCHS = 10000
def generate_gif(gif_name='mnist_gan.gif', pattern='image*.png'):

    with imageio.get_writer(gif_name, mode='I') as writer:

        filenames = glob(pattern)

        filenames = sorted(filenames)

        last = -1

        for i,filename in enumerate(filenames):

            frame = 2*(i**0.5)

            if round(frame) > round(last):

                last = frame

            else:

                continue

            image = imageio.imread(filename)

            writer.append_data(image)

            

        image = imageio.imread(filename)

        writer.append_data(image)



    # this is a hack to display the gif inside the notebook

    os.system('cp {} {}.png'.format(gif_name, gif_name))
def sample_images(generator, epoch, save=True, name='mnist'):

    """ Sample images from generator, plot them and save as png"""



    noise = np.random.normal(size=(5 * 5, LATENT_DIM))

    gen_imgs = generator.predict(noise)

    gen_imgs = 0.5 * gen_imgs + 0.5     # Rescale images 0-1



    fig, axs = plt.subplots(5, 5)

    c = 0

    for i in range(5):

        for j in range(5):

            axs[i,j].imshow(gen_imgs[c, :,:,0], cmap='gray')

            axs[i,j].axis('off')

            c += 1

            

    if save:

        fig.savefig("{}_{}.png".format(name, epoch))

        plt.close()

    else:

        plt.show()
(X, _), (_, _) = kr.datasets.mnist.load_data()



X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')

X = (X - 127.5) / 127.5 # Normalize the images to [-1, 1]
def build_generator(output_shape=(28, 28, 1)): 

    model = kr.Sequential(name='generator')

    

    model.add(kr.layers.Dense(256, input_shape=(LATENT_DIM, )))

    model.add(kr.layers.LeakyReLU(alpha=0.2))

    model.add(kr.layers.BatchNormalization(momentum=0.8))



    model.add(kr.layers.Dense(512))

    model.add(kr.layers.LeakyReLU(alpha=0.2))

    model.add(kr.layers.BatchNormalization(momentum=0.8))



    model.add(kr.layers.Dense(1024))

    model.add(kr.layers.LeakyReLU(alpha=0.2))

    model.add(kr.layers.BatchNormalization(momentum=0.8))



    model.add(kr.layers.Dense(np.prod(output_shape), activation='tanh'))

    model.add(kr.layers.Reshape(output_shape))



    return model





generator = build_generator()

generator.summary()
def build_discriminator(input_shape=(28, 28, 1)):

    model = kr.Sequential(name='discriminator')



    model.add(kr.layers.Flatten(input_shape=input_shape))

    model.add(kr.layers.Dense(512))

    model.add(kr.layers.LeakyReLU(alpha=0.2))



    model.add(kr.layers.Dense(256))

    model.add(kr.layers.LeakyReLU(alpha=0.2))



    model.add(kr.layers.Dense(1, activation='sigmoid'))



    return model





discriminator = build_discriminator()

discriminator.summary()
optimizer = kr.optimizers.Adam(0.0002, 0.5)



discriminator.compile(loss='binary_crossentropy', optimizer=optimizer,  metrics=['acc'])

discriminator.trainable = False    # For GAN we will only train the generator



z = kr.Input(shape=(LATENT_DIM,)) 

valid = discriminator(generator(z))



model = kr.Model(z, valid)

model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.summary()
# Adversarial ground truths

valid_labels = np.ones((BATCH_SIZE, 1))

fake_labels = np.zeros((BATCH_SIZE, 1))



for epoch in range(EPOCHS):

    noise = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))

    

    # Shuffle and batch data 

    imgs = X[np.random.randint(0, X.shape[0], BATCH_SIZE)] 

    

    loss_real = discriminator.train_on_batch(imgs, valid_labels)

    loss_fake = discriminator.train_on_batch(generator.predict(noise), fake_labels)

    d_loss, d_acc = 0.5 * np.add(loss_real, loss_fake)

    

    noise = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))

    g_loss = model.train_on_batch(noise, valid_labels)

    display.clear_output(wait=True)

    print ("Epoch : %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, 100*d_acc, g_loss))



    # If at save interval => save generated image samples

    if epoch % SAMPLE_INTERVAL == 0:

        sample_images(generator, epoch, name='../working/mnist')
sample_images(generator, None, save=False)
generate_gif(gif_name='../working/mnist_gan.gif', pattern='../working/mnist*.png')

display.Image(filename="../working/mnist_gan.gif.png")
(X, _), (_, _) = kr.datasets.fashion_mnist.load_data()



X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')

X = (X - 127.5) / 127.5 # Normalize the images to [-1, 1]
def generator_model():

    model = kr.Sequential()

    

    model.add(kr.layers.Dense(7 * 7 * 128, activation="relu", input_shape=(LATENT_DIM,)))

    model.add(kr.layers.Reshape((7, 7, 128)))

    

    model.add(kr.layers.UpSampling2D())

    model.add(kr.layers.Conv2D(128, (3, 3), padding='same'))

    model.add(kr.layers.BatchNormalization(momentum=0.8))

    model.add(kr.layers.ReLU())



    model.add(kr.layers.UpSampling2D())

    model.add(kr.layers.Conv2D(64, (3, 3), padding='same'))

    model.add(kr.layers.BatchNormalization(momentum=0.8))

    model.add(kr.layers.ReLU())



    model.add(kr.layers.Conv2D(1, (3, 3), padding='same', activation='tanh'))

  

    return model





generator = generator_model()

generator.summary()
def discriminator_model():

    model = kr.Sequential()

    

    model.add(kr.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))

    model.add(kr.layers.LeakyReLU(alpha=0.2))

    model.add(kr.layers.Dropout(0.25))

      

    model.add(kr.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))

    # model.add(kr.layers.ZeroPadding2D(padding=((0,1),(0,1))))

    model.add(kr.layers.BatchNormalization(momentum=0.8))

    model.add(kr.layers.LeakyReLU(alpha=0.2))

    model.add(kr.layers.Dropout(0.25))

    

    model.add(kr.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))

    model.add(kr.layers.BatchNormalization(momentum=0.8))

    model.add(kr.layers.LeakyReLU(alpha=0.2))

    model.add(kr.layers.Dropout(0.25))

    

    model.add(kr.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))

    model.add(kr.layers.BatchNormalization(momentum=0.8))

    model.add(kr.layers.LeakyReLU(alpha=0.2))

    model.add(kr.layers.Dropout(0.25))

       

    model.add(kr.layers.Flatten())

    model.add(kr.layers.Dense(1, activation='sigmoid'))

     

    return model





discriminator = discriminator_model()

discriminator.summary()
optimizer = kr.optimizers.Adam(0.0002, 0.5)



discriminator.compile(loss='binary_crossentropy', optimizer=optimizer,  metrics=['acc'])

discriminator.trainable = False    # For GAN we will only train the generator



z = kr.Input(shape=(LATENT_DIM,)) 

valid = discriminator(generator(z))



model = kr.Model(z, valid)

model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.summary()
# Adversarial ground truths

valid_labels = np.ones((BATCH_SIZE, 1))

fake_labels = np.zeros((BATCH_SIZE, 1))



for epoch in range(EPOCHS):

    noise = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))

    

    # Shuffle and batch data 

    imgs = X[np.random.randint(0, X.shape[0], BATCH_SIZE)] 

    

    loss_real = discriminator.train_on_batch(imgs, valid_labels)

    loss_fake = discriminator.train_on_batch(generator.predict(noise), fake_labels)

    d_loss, d_acc = 0.5 * np.add(loss_real, loss_fake)

    

    noise = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))

    g_loss = model.train_on_batch(noise, valid_labels)

    display.clear_output(wait=True)

    print ("Epoch : %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, 100*d_acc, g_loss))



    # If at save interval => save generated image samples

    if epoch % SAMPLE_INTERVAL == 0:

        sample_images(generator, epoch, name='../working/fmnist')
sample_images(generator, None, save=False, name='../working/fmnist')
generate_gif(gif_name='../working/fmnist_gan.gif', pattern='../working/fmnist*.png')

display.Image(filename="../working/fmnist_gan.gif.png")