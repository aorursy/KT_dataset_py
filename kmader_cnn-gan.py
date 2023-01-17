%matplotlib inline

import os,random

import numpy as np



from keras.utils import np_utils

import keras.models as models

from keras.layers import Input, BatchNormalization

from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten

from keras.layers.advanced_activations import LeakyReLU



from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconv2D, UpSampling2D



from keras.optimizers import Adam

import matplotlib.pyplot as plt

import seaborn as sns

import random, sys, keras

from keras.models import Model

from IPython import display



from keras.utils import np_utils

from tqdm import tqdm

from keras import backend as K

from keras.utils.io_utils import HDF5Matrix

def make_trainable(net, val):

    net.trainable = val

    for l in net.layers:

        l.trainable = val



K.set_image_dim_ordering('tf')
base_path = os.path.join('..', 'input')

train_h5_path = os.path.join(base_path, 'food_c101_n10099_r32x32x1.h5')

test_h5_path = os.path.join(base_path, 'food_test_c101_n1000_r32x32x1.h5')
X_train = HDF5Matrix(train_h5_path, 'images')[:]

y_train = HDF5Matrix(train_h5_path, 'category')

print('In Data',X_train.shape, 'min',X_train[0].min(),'max', X_train[0].max(),'=>', y_train.shape)
X_test = HDF5Matrix(test_h5_path, 'images')[:]

y_test = HDF5Matrix(test_h5_path, 'category')

print('In Data',X_test.shape,'=>', y_test.shape)

n_cat = y_test.shape[1]
def plot_real(n_ex=16,dim=(4,4), figsize=(10,10) ):

    

    idx = np.random.randint(0,X_train.shape[0],n_ex)

    generated_images = X_train[idx,:,:,:]



    plt.figure(figsize=figsize)

    for i in range(generated_images.shape[0]):

        plt.subplot(dim[0],dim[1],i+1)

        img = generated_images[i,:,:,0]

        plt.imshow(img, cmap = 'bone')

        plt.axis('off')

    plt.tight_layout()

    plt.show()

plot_real()
shp = X_train.shape[1:]



dropout_rate = 0.25

# Optim



opt = Adam(lr=1e-3)

dopt = Adam(lr=1e-4)


# Build Generative model ...

nch = n_cat//1 # use the categories as a baseline

g_input = Input(shape=[n_cat])

H = Dense(nch*8*8, kernel_initializer='glorot_normal', activation = 'linear')(g_input)

H = BatchNormalization()(H)

H = LeakyReLU(0.2)(H)

H = Reshape( [8, 8, nch] )(H)

H = Deconv2D(nch//2, strides = (2, 2), kernel_size = (2,2), activation = 'linear')(H)

H = LeakyReLU(0.2)(H)

H = Convolution2D(nch//2, (3, 3), padding='same', kernel_initializer='glorot_uniform', activation = 'linear')(H)

H = BatchNormalization()(H)

H = LeakyReLU(0.2)(H)

H = Deconv2D(nch//4, strides = (2, 2), kernel_size = (2,2), activation = 'linear')(H)

H = LeakyReLU(0.2)(H)

H = Convolution2D(nch//4, (3, 3), padding='same', kernel_initializer='glorot_uniform', activation = 'linear')(H)

H = BatchNormalization()(H)

H = LeakyReLU(0.2)(H)

H = Convolution2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform', activation = 'linear')(H)

g_V = Activation('sigmoid')(H)

generator = Model(g_input, g_V, name = 'GeneratorModel')

generator.compile(loss='binary_crossentropy', optimizer=opt)

generator.summary()
# Build Discriminative model ...

d_input = Input(shape=shp)

H = Convolution2D(256, (3, 3), strides=(2, 2), padding = 'valid', activation='linear')(d_input)

H = LeakyReLU(0.2)(H)

H = Dropout(dropout_rate)(H)

H = Convolution2D(512, (3, 3), strides=(2, 2), padding = 'valid', activation='linear')(H)

H = LeakyReLU(0.2)(H)

H = Dropout(dropout_rate)(H)

H = Convolution2D(1024, (3, 3), strides=(2, 2), padding = 'valid', activation='linear')(H)

H = LeakyReLU(0.2)(H)

H = Dropout(dropout_rate)(H)

H = Flatten()(H)

H = Dense(256, activation = 'linear')(H)

H = LeakyReLU(0.2)(H)

H = Dropout(dropout_rate)(H)

d_V = Dense(2, activation='softmax')(H)

discriminator = Model(d_input,d_V, name = 'DiscriminatorModel')

discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)

discriminator.summary()
make_trainable(discriminator, False)

# Build stacked GAN model

gan_input = Input(shape=[n_cat])

H = generator(gan_input)

gan_V = discriminator(H)

GAN = Model(gan_input, gan_V)

GAN.compile(loss='categorical_crossentropy', optimizer=opt)

GAN.summary()
def plot_gen(n_ex=9,dim=(3,3), figsize=(10,10) ):

    noise = np.random.uniform(0,1,size=[n_ex,n_cat])

    generated_images = generator.predict(noise, batch_size = 8)



    plt.figure(figsize=figsize)

    for i in range(generated_images.shape[0]):

        plt.subplot(dim[0],dim[1],i+1)

        img = generated_images[i,:,:,0]

        plt.imshow(img, cmap = 'bone')

        plt.axis('off')

    plt.tight_layout()

    plt.show()

plot_gen()
ntrain = 500

XT = X_train[:ntrain][:,:,:]



# Pre-train the discriminator network ...

noise_gen = np.random.uniform(0,1,size=[XT.shape[0],n_cat])

generated_images = generator.predict(noise_gen, batch_size = 8)

X = np.concatenate((XT, generated_images))

n = XT.shape[0]

y = np.zeros([2*n,2])

y[:n,1] = 1

y[n:,0] = 1
make_trainable(discriminator,True)

discriminator.fit(X,y, epochs=1, batch_size = 4, shuffle = True)

y_hat = discriminator.predict(X, batch_size = 8)
y_hat_idx = np.argmax(y_hat,axis=1)

y_idx = np.argmax(y,axis=1)

diff = y_idx-y_hat_idx

n_tot = y.shape[0]

n_rig = (diff==0).sum()

acc = n_rig*100.0/n_tot

print("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))
# set up loss storage vector

losses = {"d":[], "g":[]}

def plot_loss(losses):

        display.clear_output(wait=True)

        display.display(plt.gcf())

        plt.figure(figsize=(10,8))

        plt.plot(losses["d"], label='discriminitive loss')

        plt.plot(losses["g"], label='generative loss')

        plt.legend()

        plt.show()
def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):  

        

        # Make generative images

        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]    

        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,n_cat])

        generated_images = generator.predict(noise_gen)

        

        # Train discriminator on generated images

        X = np.concatenate((image_batch, generated_images))

        y = np.zeros([2*BATCH_SIZE,2])

        y[0:BATCH_SIZE,1] = 1

        y[BATCH_SIZE:,0] = 1

        

        make_trainable(discriminator,True)

        d_loss  = discriminator.train_on_batch(X,y)

        losses["d"].append(d_loss)

    

        # train Generator-Discriminator stack on input noise to non-generated output class

        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,n_cat])

        y2 = np.zeros([BATCH_SIZE,2])

        y2[:,1] = 1

        

        make_trainable(discriminator,False)

        g_loss = GAN.train_on_batch(noise_tr, y2 )

        losses["g"].append(g_loss)

        

        # Updates plots

        if e%plt_frq==plt_frq-1:

            plot_loss(losses)

            plot_gen()

        
train_for_n(nb_epoch=400, plt_frq=25,BATCH_SIZE=4)
K.set_value(opt.lr, 1e-4)

K.set_value(dopt.lr, 1e-5)

train_for_n(nb_epoch=100, plt_frq=100,BATCH_SIZE=4)
K.set_value(opt.lr, 1e-5)

K.set_value(dopt.lr, 1e-6)

train_for_n(nb_epoch=100, plt_frq=300,BATCH_SIZE=4)
plot_loss(losses)
plot_gen(25,(5,5),(12,12))
plot_real()