# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





from __future__ import absolute_import

from __future__ import print_function

from keras.utils import np_utils # For y values



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





import os

print(os.listdir("../input"))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math





# For plotting

%matplotlib inline

import seaborn as sns

# For Keras

from tensorflow.keras.layers import Activation, Dense, Input

from tensorflow.keras.layers import Conv2D, Flatten

from tensorflow.keras.layers import Reshape, Conv2DTranspose

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.models import Model

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import load_model

from tensorflow.keras.utils import plot_model
os.listdir("../input/mnist-in-csv")
train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
train.head()
#Remove the first column from the data, as it is the label and put the rest in X

X_train = train.iloc[:, 1:].values#.reshape(-1,28,28,1)

#Remove everything except the first column from the data, as it is the label and put it in y

y_train = train.iloc[:, :1].values
#Remove the first column from the data, as it is the label and put the rest in X

X_test = test.iloc[:, 1:].values/255.#.reshape(-1,28,28,1)

#Remove everything except the first column from the data, as it is the label and put it in y

y_test = test.iloc[:, :1].values
X_train.shape
def build_generator(inputs, image_size): # This function builds a generator for DCGAN

    image_resize = image_size // 4

    kernel_size = 5

    layer_filters = [128, 64, 32, 1]

    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)

    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:

        if filters > layer_filters[-2]:

            strides = 2

        else:

            strides = 1

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=strides,padding='same')(x)

    x = Activation('sigmoid')(x)

    generator = Model(inputs, x, name='generator')

    return generator
def build_discriminator(inputs): #This function builds a discriminator

    kernel_size = 5

    layer_filters = [32, 64, 128, 256]

    x = inputs

    for filters in layer_filters:

        if filters == layer_filters[-1]:

            strides = 1

        else:

            strides = 2

        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(filters=filters,

                   kernel_size=kernel_size,

                   strides=strides,

                   padding='same')(x)



    x = Flatten()(x)

    x = Dense(1)(x)

    x = Activation('sigmoid')(x)

    discriminator = Model(inputs, x, name='discriminator')

    return discriminator
def train(models, x_train, params):

    generator, discriminator, adversarial = models 

    batch_size, latent_size, train_steps, model_name = params 

    save_interval = 500 

    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size]) 

    train_size = x_train.shape[0] 

    for i in range(train_steps):

        rand_indexes = np.random.randint(0, train_size, size=batch_size)

        real_images = x_train[rand_indexes]

        noise = np.random.uniform(-1.0,1.0,size=[batch_size, latent_size])

        fake_images = generator.predict(noise)

        x = np.concatenate((real_images, fake_images))

        y = np.ones([2 * batch_size, 1])

        y[batch_size:, :] = 0.0

        loss, acc = discriminator.train_on_batch(x, y)

        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        noise = np.random.uniform(-1.0,1.0, size=[batch_size, latent_size])

        y = np.ones([batch_size, 1])

        loss, acc = adversarial.train_on_batch(noise, y)

        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)

        print(log)

        if (i + 1) % save_interval == 0:

            plot_images(generator,noise_input=noise_input,show=False,step=(i + 1),model_name=model_name)

        if i == 500 or (i+1) % 10000 == 0:

            generator.save(model_name + "_" + str(i) + "_" + ".h5")

    generator.save(model_name + ".h5")
def plot_images(generator,noise_input,show=False,step=0,model_name="gan"):

    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "%05d.png" % step)

    images = generator.predict(noise_input)

    plt.figure(figsize=(2.2, 2.2))

    num_images = images.shape[0]

    image_size = images.shape[1]

    rows = int(math.sqrt(noise_input.shape[0]))

    for i in range(num_images):

        plt.subplot(rows, rows, i + 1)

        image = np.reshape(images[i], [image_size, image_size])

        plt.imshow(image, cmap='gray')

        plt.axis('off')

    plt.savefig(filename)

    if show:

        plt.show()

    else:

        plt.close('all')
# reshape data for CNN as (28, 28, 1) and normalize

image_size = 28

X_train = np.reshape(X_train, [-1, 28, 28, 1])

X_train = X_train.astype('float32') / 255
model_name = "dcgan_mnist"

# network parameters

# the latent or z vector is 100-dim

latent_size = 100

batch_size = 64

train_steps = 40000

lr = 2e-4

decay = 6e-8

input_shape = (image_size, image_size, 1)
# build discriminator model

inputs = Input(shape=input_shape, name='discriminator_input') # Input Layer TF

discriminator = build_discriminator(inputs)

optimizer = RMSprop(lr=lr, decay=decay)

discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

#discriminator.summary()
plot_model(discriminator)
# build generator model

input_shape = (latent_size, )

inputs = Input(shape=input_shape, name='z_input')

generator = build_generator(inputs, image_size)

#generator.summary()
plot_model(generator)
plt.imshow(

    generator.predict(np.random.uniform(-1.0,1.0,size=[1, 100])).reshape(28,28)

)
# build adversarial model

optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)

# freeze the weights of discriminator during adversarial training

discriminator.trainable = False

# adversarial = generator + discriminator

adversarial = Model(inputs, discriminator(generator(inputs)),name=model_name)

adversarial.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

#adversarial.summary()
plot_model(adversarial)
# train discriminator and adversarial networks

models = (generator, discriminator, adversarial)

params = (batch_size, latent_size, train_steps, model_name)
#train(models, X_train, params)
def test_generator(generator, K):

    noise_input = np.random.uniform(-1.0, 1.0, size=[K, 100])

    plot_images(generator,noise_input=noise_input,show=True,model_name="test_outputs")
#generator.load_weights('dcgan_mnist_19999_.h5')
#test_generator(generator, 1)