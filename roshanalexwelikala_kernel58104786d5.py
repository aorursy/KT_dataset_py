# example of a dcgan on dogs

from numpy import expand_dims

from numpy import zeros

from numpy import ones

from numpy import vstack

from numpy.random import randn

from numpy.random import randint

from keras.datasets.cifar10 import load_data

from keras.optimizers import Adam

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Reshape

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import Conv2DTranspose

from keras.layers import LeakyReLU

from keras.layers import Dropout

from keras.layers import BatchNormalization

from matplotlib import pyplot



import os

import numpy as np

from PIL import Image

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

import xml.etree.ElementTree as et

import tensorflow as tf

print(tf.__version__)

import shutil



# Batch norm layers are recommended in both the discriminator and generator models, except the output of the generator

# and input to the discriminator (from DCGAN paper). I found they don't work when momentum at default of 0.99, hence I

# set to 0.8.



# In GANs, the recommendation is to not use pooling layers, and instead use the stride. Already in brownlee implementation



# define the standalone discriminator model

def define_discriminator(in_shape=(64,64,3)):

    model = Sequential()

    # normal

    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))

    model.add(LeakyReLU(alpha=0.2))

    # downsample

    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))

    # model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    # downsample

    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))

    # model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    # downsample

    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))

    # model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    # classifier

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

    # compile model

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model
# define the standalone generator model

def define_generator(latent_dim):

    model = Sequential()

    # foundation for 8x8 image

    n_nodes = 256 * 8 * 8

    model.add(Dense(n_nodes, input_dim=latent_dim))

    # model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((8, 8, 256)))

    # upsample to 16x16

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

    # model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    # upsample to 32x32

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

    # model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    # upsample to 64x64

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

    # model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    # output layer

    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))

    model.summary()

    return model
# define the combined generator and discriminator model, for updating the generator

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

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt)

    model.summary()

    return model
def read_image(file, bounds):

    image = open_image(file, bounds)

    image = normalize_image(image)

    return image





def open_image(file, bounds):

    image = Image.open(file)

    image = image.crop(bounds)

    image = image.resize((64, 64))

    return np.array(image)





# Normalization, [-1,1] Range

def normalize_image(image):

    image = np.asarray(image, np.float32)

    image = image / 127.5 - 1

    return img_to_array(image)



# Restore, [0,255] Range

def denormalize_image(image):

    return ((image+1)*127.5).astype(np.uint8)





def load_images():

    images = []



    for breed in os.listdir('../input/annotation/Annotation/'):

        for dog in os.listdir('../input/annotation/Annotation/' + breed):

            tree = et.parse('../input/annotation/Annotation/' + breed + '/' + dog)

            root = tree.getroot()

            objects = root.findall('object')

            for o in objects:

                box = o.find('bndbox')

                xmin = int(box.find('xmin').text)

                ymin = int(box.find('ymin').text)

                xmax = int(box.find('xmax').text)

                ymax = int(box.find('ymax').text)



            bounds = (xmin, ymin, xmax, ymax)

            try:

                image = read_image('../input/all-dogs/all-dogs/' + dog + '.jpg', bounds)

                images.append(image)

            except:

                print('No image', dog)



    return np.array(images)


# select real samples

def generate_real_samples(dataset, n_samples):

    # choose random instances

    ix = randint(0, dataset.shape[0], n_samples)

    # retrieve selected images

    X = dataset[ix]

    # generate 'real' class labels (1)

    y = ones((n_samples, 1))

    return X, y



# generate points in latent space as input for the generator

def generate_latent_points(latent_dim, n_samples):

    # generate points in the latent space

    x_input = randn(latent_dim * n_samples)

    # reshape into a batch of inputs for the network

    x_input = x_input.reshape(n_samples, latent_dim)

    return x_input



# use the generator to generate n fake examples, with class labels

def generate_fake_samples(g_model, latent_dim, n_samples):

    # generate points in latent space

    x_input = generate_latent_points(latent_dim, n_samples)

    # predict outputs

    X = g_model.predict(x_input)

    # create 'fake' class labels (0)

    y = zeros((n_samples, 1))

    return X, y



# create and save a plot of generated images

def save_plot(examples, epoch, n=7):

    # scale from [-1,1] to [0,1]

    examples = (examples + 1) / 2.0

    # plot images

    for i in range(n * n):

        # define subplot

        pyplot.subplot(n, n, 1 + i)

        # turn off axis

        pyplot.axis('off')

        # plot raw pixel data

        pyplot.imshow(examples[i])

    # save plot to file

    # filename = 'generated_plot_e%03d.png' % (epoch+1)

    # pyplot.savefig(filename)

    # pyplot.close()

    pyplot.show()
# evaluate the discriminator, plot generated images, save generator model

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):

    # prepare real samples

    X_real, y_real = generate_real_samples(dataset, n_samples)

    # evaluate discriminator on real examples

    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

    # prepare fake examples

    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

    # evaluate discriminator on fake examples

    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    # summarize discriminator performance

    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))

    # save plot

    save_plot(x_fake, epoch)

    # save the generator model tile file

    # filename = 'generator_model_%03d.h5' % (epoch+1)

    # g_model.save(filename)

# train the generator and discriminator

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=500, n_batch=128):

    bat_per_epo = int(dataset.shape[0] / n_batch)

    half_batch = int(n_batch / 2)

    # manually enumerate epochs

    for i in range(n_epochs):

        # enumerate batches over the training set

        for j in range(bat_per_epo):

            # get randomly selected 'real' samples

            X_real, y_real = generate_real_samples(dataset, half_batch)

            # update discriminator model weights

            d_loss1, _ = d_model.train_on_batch(X_real, y_real)

            # generate 'fake' examples

            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            # update discriminator model weights

            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            # prepare points in latent space as input for the generator

            X_gan = generate_latent_points(latent_dim, n_batch)

            # create inverted labels for the fake samples

            y_gan = ones((n_batch, 1))

            # update the generator via the discriminator's error

            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            # summarize loss on this batch

            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %

                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

        # evaluate the model performance, sometimes

        if (i+1) % 1 == 0:

            summarize_performance(i, g_model, d_model, dataset, latent_dim)
# size of the latent space

latent_dim = 100

# create the discriminator

d_model = define_discriminator()

# create the generator

g_model = define_generator(latent_dim)

# create the gan

gan_model = define_gan(g_model, d_model)

# load image data

# dataset = load_real_samples()

dataset = load_images()

# train model

train(g_model, d_model, gan_model, dataset, latent_dim)
# Submission

def save_images(generator):

    if not os.path.exists('../output'):

        os.mkdir('../output')



    latent_points = generate_latent_points(100, 10000)

    generated_images = generator.predict(latent_points)



    for i in range(generated_images.shape[0]):

        image = denormalize_image(generated_images[i])

        image = array_to_img(image)

        image.save('../output/' + str(i) + '.png')



    shutil.make_archive('images', 'zip', '../output')





save_images(g_model)