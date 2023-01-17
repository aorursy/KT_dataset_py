# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: |https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
a=np.arange(68)
np.random.shuffle(a)
print(a)
#code for shuffling minibatches
i=0
j=0
while i<68:
    Y=[]
    while j<min(i+10,68):
        X=np.load("/kaggle/input/minibatches/minibatch%d.npz" % (a[j]))
        X=X['arr_0']
        j=j+1
        Y.append(X)
    Y1=np.vstack(Y)
    np.random.shuffle(Y1)
    Y1=np.split(Y1,Y1.shape[0]/16)
    for index,arr in enumerate(Y1):
        np.savez("/kaggle/working/shuffled_minibatch%d.npz"%(i+index),arr)
    i=j
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from tensorflow import keras
from tensorflow.keras.layers import (Input, Activation,
                                     BatchNormalization, Conv3D,
                                     LeakyReLU, Conv3DTranspose)
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K


def AutoEncoderModel():
    
    # encoder
    X_input = Input((16, 128, 128, 3))
    
    model = Sequential()
    model.add(Conv3D(32, 3, padding='same',input_shape=(16,128,128,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
    
    # current shape is 8x64x64x32
    
    model.add(Conv3D(48, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
    
    # current shape is 4x32x32x48
    
    model.add(Conv3D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid'))
    
    # current shape is 2x16x16x64
    
    model.add(Conv3D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same'))
    
    # current shape is 2x16x16x64
    
    #####################################
    # decoder

    model.add(Conv3DTranspose(48, 2, strides=(2, 2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    # current shape is 4x32x32x48
    
    model.add(Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    # current shape is 8x64x64x32
    
    model.add(Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    # current shape is 16x128x128x32
    
    model.add(Conv3D(3, 3, strides=(1, 1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    
    # current shape is 16x128x128x3

    # model = Model(inputs=X_input, outputs=X, name='AutoEncoderModel')
    return model


def custom_loss(new, original):
    reconstruction_error = K.mean(K.square(new-original))
    return reconstruction_error

autoEncoderModel = AutoEncoderModel()
opt = keras.optimizers.Adam(lr=0.001)
autoEncoderModel.compile(
    loss=custom_loss, optimizer=opt, metrics=['accuracy'])
print(autoEncoderModel.summary())
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras import Sequential
def create_discriminator_model():

    X_input = Input((16, 128, 128, 3))

    # not sure about the axis in batch norm
    # do we also add dropout after batchnorm/pooling?

    # Convolutional Layers
    # changed the no of filters
    model= Sequential()
    model.add(Conv3D(filters=48, kernel_size=(2, 2, 2), padding="same",input_shape=(16, 128, 128, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(filters=64, kernel_size=(2, 2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(filters=128, kernel_size=(2, 2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(filters=128, kernel_size=(2, 2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    # to add the 5th layer change the cap to 32 frames

    # X=Conv3D(filters=256,kernel_size=(2,2,2),padding="same")(X)
    # X=BatchNormalization()(X)
    # X=Activation('relu')(X)
    # X=MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(X)

    # Fully connected layers

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    # add batch norm to dense layer
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())
    # activation done with loss fn
    # for numerical stability
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = create_discriminator_model()
opt = keras.optimizers.Adam(lr=0.001)
loss = BinaryCrossentropy()
discriminator.compile(loss=loss,
                      optimizer=opt,
                      metrics=['accuracy'])
print(discriminator.summary())

import tensorflow as tf
class GAN():
    def __init__(self):
        self.image_shape=(16,128,128,3)
        learning_rate=0.03
        opt=keras.optimizers.Adam(lr=learning_rate)
        opt1=keras.optimizers.Adam(lr=learning_rate)
        opt_slow=keras.optimizers.Adam(lr=10*learning_rate)
        #Build and compile the discriminator
        self.discriminator=create_discriminator_model()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        #Build and compile the generator
        self.generator=AutoEncoderModel()
        self.generator.compile(loss='mse',optimizer=opt_slow)

        #the generator takes a video as input and generates a modified video
        z = Input(shape=(self.image_shape))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=opt1,metrics=['accuracy'])

    def train(self,epochs,mini_batch_size):
        #this function will need to be added later
        for epoch in range(epochs):
            d_loss_sum=np.zeros(2)
            reconstruct_error_sum=0
            g_loss_sum=np.zeros(2)
            for i in range(68):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                minibatch=np.load('/kaggle/working/shuffled_minibatch%d.npz' %(i))
                minibatch=minibatch['arr_0']
                minibatch=K.cast(minibatch,'float32')
                #normalize inputs
                minibatch/=255
                gen_vids=self.generator.predict(minibatch)
                #might have to combine these to improve batch norm
                d_loss_real=self.discriminator.train_on_batch(minibatch,np.ones((mini_batch_size,1)))
                d_loss_fake=self.discriminator.train_on_batch(gen_vids,np.zeros((mini_batch_size,1)))
                d_loss=0.5*np.add(d_loss_real,d_loss_fake)
                # ---------------------
                #  Train Generator
                # ---------------------
                # The generator wants the discriminator to label the generated samples as valid (ones)
                valid_y = np.array([1] * mini_batch_size)
                # Train the generator
                g_loss = self.combined.train_on_batch(minibatch,valid_y)
                reconstruct_error=self.generator.train_on_batch(minibatch,minibatch)
                d_loss_sum+=d_loss
                g_loss_sum+=g_loss
                reconstruct_error_sum+=reconstruct_error
            g_loss=g_loss_sum/68
            d_loss=d_loss_sum/68
            reconstruct_error=reconstruct_error_sum/68
            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, accuracy %.2f%% from which %f is combined loss and %f is reconstruction loss]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]+reconstruct_error,g_loss[1]*100,g_loss[0],reconstruct_error))
        
gan = GAN()
print(gan.combined.summary())

# print(gan.discriminator.summary())
# print(gan.generator.summary())

gan.train(100,16)
