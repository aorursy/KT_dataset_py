# Imports

import numpy as np 

import pandas as pd 



from keras.models import Model

from keras.layers import Flatten, Conv2D, MaxPool2D, Input

from keras.preprocessing.image import ImageDataGenerator



import os
# variables

WIDTH = 256

HEIGTH = 512

LATENT_DIM = 16

IMAGE_FOLDER =  '../input/images'
# Create Encoder

def Encoder(input_shape=(HEIGTH,WIDTH), latent_dim=16): 

    input_img= Input(shape=(512, 256, 3)) 

    x = Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(HEIGTH, WIDTH, 3),padding='same')(input_img)

    x = MaxPool2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same')(x)

    x = MaxPool2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(8, kernel_size=(3, 3), activation='relu',padding='same')(x)

    

    encoded = MaxPool2D(pool_size=(2, 2),padding='same')(x)

    return Model(input_img, encoded)
train_datagen = ImageDataGenerator(

        rescale=1./255)



train_generator = train_datagen.flow_from_directory(IMAGE_FOLDER,

        target_size=(HEIGTH,WIDTH),

        color_mode='rgb',

        class_mode='input')
encoder = Encoder((HEIGTH,WIDTH), LATENT_DIM)

x= Input(shape=(HEIGTH,WIDTH,3))

#x= Input(shape=(64,32,8))

encoder.summary()

autoencoder = Model(x, encoder(x))



autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')



autoencoder.summary()

autoencoder.fit_generator(

        train_generator,

        steps_per_epoch=2000,

        epochs=50)
#save encoding as numpy array

#np.savetxt("encodings.csv", Flatten(encoder.predict(x)), delimiter=",")