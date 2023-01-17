# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#for (root, dirs, files) in os.walk('../input'):

#    print(root, dirs)

width = 64

height = 64

channel = 3

dogs = []

epochs = 200

batch = 128

log_interval = 20
import numpy as np

from PIL import Image

#import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET



for breed in os.listdir('../input/annotation/Annotation'):

    for dog in (os.listdir('../input/annotation/Annotation/'+breed)):

        annot = '../input/annotation/Annotation/'+breed+'/'+dog

        fullyQualifiedDog = '../input/all-dogs/all-dogs/'+dog

        try: img = Image.open(fullyQualifiedDog+'.jpg')

        except: continue

       

        file = open(annot, 'r')

        root = ET.parse(file).getroot()

        

        for obj in root.findall('object'):

            bndbox = obj.find('bndbox')

            xmin = bndbox.find('xmin').text

            xmax = bndbox.find('xmax').text

            ymin = bndbox.find('ymin').text

            ymax = bndbox.find('ymax').text

            cropImg = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))

            resizedImg = cropImg.resize((width,height), Image.ANTIALIAS)

            dogs.append(np.asarray(resizedImg))

dogs = np.asarray(dogs)

print(dogs.shape)
from keras.models import Sequential

from keras.layers import Dense, BatchNormalization, Reshape, Flatten, Activation, Dropout, Conv2D, Conv2DTranspose

from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam

from keras.initializers import RandomNormal





def conv_generator():

    model = Sequential()

    model.add(Dense(units=4*4*512, input_shape=(4096,), kernel_initializer=RandomNormal(stddev=0.02, mean=0.0)))

    model.add(Reshape(target_shape=(4,4,512)))

    model.add(BatchNormalization(momentum=0.5))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Conv2DTranspose(filters=512, kernel_size=(3,3), strides=(2,2), padding='same', data_format='channels_last', kernel_initializer=RandomNormal(stddev=0.02, mean=0.0)))

    model.add(BatchNormalization(momentum=0.5))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(2,2), padding='same', data_format='channels_last', kernel_initializer=RandomNormal(stddev=0.02, mean=0.0)))

    model.add(BatchNormalization(momentum=0.5))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', data_format='channels_last', kernel_initializer=RandomNormal(stddev=0.02, mean=0.0)))

    model.add(BatchNormalization(momentum=0.5))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Conv2DTranspose(filters=3, kernel_size=(3,3), strides=(2,2), padding='same', data_format='channels_last', kernel_initializer=RandomNormal(stddev=0.02, mean=0.0)))

    model.add(Activation('tanh'))

    return model



def generator():

    model = Sequential()

    n_nodes = 256

    model.add(Dense(n_nodes, input_shape=(100,)))

    model.add(LeakyReLU(alpha=0.2))

    model.add(BatchNormalization())

    model.add(Dense(512))

    model.add(LeakyReLU(alpha=0.2))

    model.add(BatchNormalization())

    model.add(Dense(1024))

    model.add(LeakyReLU(alpha=0.2))

    model.add(BatchNormalization())

    model.add(Dense(width*height*channel, activation='tanh'))

    model.add(Reshape((width, height, channel)))

    return model



def discriminator():

    model = Sequential()

    model.add(Flatten(input_shape=(width, height, channel)))

    model.add(Dense((width*height*channel), input_shape=(width, height, channel)))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(int((width*height*channel)/2)))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(1, activation='sigmoid'))

    return model



def conv_discriminator():

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding='same', data_format='channels_last', kernel_initializer=RandomNormal(stddev=0.02, mean=0.0)))

    model.add(BatchNormalization(momentum=0.5))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding='same', data_format='channels_last', kernel_initializer=RandomNormal(stddev=0.02, mean=0.0)))

    model.add(BatchNormalization(momentum=0.5))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding='same', data_format='channels_last', kernel_initializer=RandomNormal(stddev=0.02, mean=0.0)))

    model.add(BatchNormalization(momentum=0.5))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.5))

    model.add(Conv2D(filters=1, kernel_size=(4,4), strides=(1,1), padding='same'))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    return model



G = conv_generator()

G.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

G.summary()



D = conv_discriminator()

D.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])



def G_D():

    D.trainable = False

    model = Sequential()

    model.add(G)

    model.add(D)

    return model



G_D = G_D()

G_D.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

G_D.summary()



import keras

import tensorflow as tf

import matplotlib.pyplot as plt





def plot_img():

    samples = 16

    noises = np.random.uniform(-1, 1, size=(samples, 4096))

    images = G.predict(noises)

    plt.figure(figsize=(8,8))

    

    for i in range(16):

        plt.subplot(4, 4, i+1)

        plt.axis('off')

        pltImg = ((images[i] + 1)*127.5).astype(np.uint8)

        plt.imshow(images[i])

    plt.show()



def train():

    batches = int(len(dogs)/batch)

    for cnt in range(epochs):

        for batchCnt in range(batches):

            for _ in range(2):

                indx = batchCnt*batch

                good_imgs = dogs[indx:indx+batch]

                good_imgs = good_imgs.astype('float32')

                good_imgs = (good_imgs - 127.5)/127.5



                gen_noise = np.random.uniform(-1, 1, size=(batch, 4096))

                keras.backend.get_session().run(tf.global_variables_initializer())

                synthetic_imgs = G.predict(gen_noise)

                

                y_real = np.ones((batch, 1))

                y_real = y_real - 0.3 + np.random.rand(batch, 1)*0.3

                y_fake = np.zeros((batch, 1))

                y_fake = y_fake + np.random.rand(batch, 1)*0.3

                D.trainable = True

                d_loss_real = D.train_on_batch(good_imgs, y_real)

                d_loss_fake = D.train_on_batch(synthetic_imgs, y_fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            D.trainable = False

            noise = np.random.uniform(-1, 1, size=(batch, 4096))

            y_mislead = y_real

            g_loss = G_D.train_on_batch(noise, y_mislead)

        print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

    plot_img()

train()