# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm

import random as rn

from PIL import Image 

import xml.etree.ElementTree as ET 



from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Activation, Dropout, BatchNormalization, Input, LeakyReLU, Reshape

from keras.models import Model

from keras.optimizers import Adam

from keras.initializers import glorot_normal
chihuahua_dir = '../input/stanford-dogs-dataset/images/Images/n02085620-Chihuahua'

japanese_spaniel_dir = '../input/stanford-dogs-dataset/images/Images/n02085782-Japanese_spaniel'

maltese_dir = '../input/stanford-dogs-dataset/images/Images/n02085936-Maltese_dog'

pekinese_dir = '../input/stanford-dogs-dataset/images/Images/n02086079-Pekinese'

shitzu_dir = '../input/stanford-dogs-dataset/images/Images/n02086240-Shih-Tzu'

blenheim_spaniel_dir = '../input/stanford-dogs-dataset/images/Images/n02086646-Blenheim_spaniel'

papillon_dir = '../input/stanford-dogs-dataset/images/Images/n02086910-papillon'

toy_terrier_dir = '../input/stanford-dogs-dataset/images/Images/n02087046-toy_terrier'

afghan_hound_dir = '../input/stanford-dogs-dataset/images/Images/n02088094-Afghan_hound'

basset_dir = '../input/stanford-dogs-dataset/images/Images/n02088238-basset'

papillon_dir = '../input/stanford-dogs-dataset/images/Images/n02086910-papillon'

beagle_dir = '../input/stanford-dogs-dataset/images/Images/n02088364-beagle'

clumber_dir = '../input/stanford-dogs-dataset/images/Images/n02101556-clumber'





X_train = []

imgsize = 64
def training_data(data_dir):

    for img in tqdm(os.listdir(data_dir)):

        path = os.path.join(data_dir,img)

        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (imgsize, imgsize))

        img = cv2.normalize(img, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)



        X_train.append(np.array(img))
training_data(chihuahua_dir)

training_data(japanese_spaniel_dir)

training_data(maltese_dir)

training_data(pekinese_dir)

training_data(shitzu_dir)

training_data(blenheim_spaniel_dir)

training_data(papillon_dir)

training_data(toy_terrier_dir)

training_data(afghan_hound_dir)

training_data(basset_dir)

training_data(papillon_dir)

training_data(beagle_dir)

training_data(clumber_dir)

X_train = np.array(X_train)

print(X_train.shape)

fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(X_train))

        ax[i,j].imshow(X_train[l])

        

plt.tight_layout()
plt.imshow(X_train[6])
img_shape = (64, 64, 3)

channels = 3

latent_dim = 100
def build_generator(latent_dim):

    z = Input(shape=(latent_dim,))

    x = Dense(4*4*512, input_dim=latent_dim)(z)

    x = Reshape((4, 4, 512))(x)

    

    x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = LeakyReLU(0.2)(x)



    

    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = LeakyReLU(0.2)(x)

    

    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = LeakyReLU(0.2)(x)

    

    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = LeakyReLU(0.2)(x)

    

    img = Conv2DTranspose(channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

    

    model = Model(z, img)

    

    return model
def build_discriminator(img_shape):

    img = Input(shape= img_shape)

    

    x = Conv2D(32, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(img)

    x = BatchNormalization()(x)

    x = LeakyReLU(0.2)(x)

    

    x = Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(0.2)(x)

    

    x = Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(0.2)(x)

    

    x = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(0.2)(x)

    

    x = Conv2D(512, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(0.2)(x)

    

    x = Flatten()(x)

    y = Dense(1, activation='sigmoid')(x)

    

    model = Model(img, y)

    return model
def build_gan(generator, discriminator):

    z = Input(shape=(latent_dim, ))

    

    img = generator(z)

    

    validity = discriminator(img)

    

    model = Model(z, validity)

    return model
discriminator = build_discriminator(img_shape)

discriminator.summary()

discriminator.compile(loss= 'binary_crossentropy', optimizer=Adam(lr=0.0005, beta_1=0.5))



discriminator.trainable = False



generator = build_generator(latent_dim)

generator.summary()

dcgan = build_gan(generator, discriminator)



dcgan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

epochs = 200

batch_size = 32

smooth = 0.1

real = np.random.normal(0.7, 0.1, (batch_size, 1))

fake = np.random.normal(0, 0.3, (batch_size, 1))



g_loss = []

d_loss = []

np.random.shuffle(X_train)



for e in range(epochs + 1):

    for i in range(len(X_train) // batch_size):

        discriminator.trainable = True

        # Real samples

        X_batch = X_train[i*batch_size:(i+1)*batch_size]

        d_loss_real = discriminator.train_on_batch(x=X_batch,

                                                   y=real)

        

        # Fake Samples

        z = np.random.normal(size=(batch_size, latent_dim))

        X_fake = generator.predict_on_batch(z)

        d_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake)

         

        # Discriminator loss

        d_loss_batch = 0.5 * (d_loss_real + d_loss_fake)

        

        discriminator.trainable = False

        # Train Generator weights

        g_loss_batch = dcgan.train_on_batch(x=z, y=real)



        print(

            'epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, i, len(X_train) // batch_size, d_loss_batch, g_loss_batch),

            100*' ',

            end='\r'

        )

    

    d_loss.append(d_loss_batch)

    g_loss.append(g_loss_batch)

    print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], g_loss[-1]), 100*' ')



    if e % 10 == 0:

        samples = 10

        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))



        for k in range(samples):

            plt.subplot(2, 5, k + 1, xticks=[], yticks=[])

            plt.imshow((x_fake[k]))



        plt.tight_layout()

        plt.show()
samples = 10

x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))

for k in range(samples):

    plt.subplot(2, 5, k + 1, xticks=[], yticks=[])

    plt.imshow((x_fake[k]))



plt.tight_layout()

plt.show()
print(X_train.shape)