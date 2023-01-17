import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

from tensorflow.keras.initializers import he_normal
def identity_block(X, f, filters, stage, block):

    """

    m -- number of examples

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    

    Returns:

    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    """

    

    # Defining name basis

    conv_name_base = f'res{stage}{block}_branch'

    bn_name_base = f'bn{stage}{block}_branch'

    

    # Retrieve filters

    F1, F2, F3 = filters

    

    # Save the input value. You'll need this later to add back to the main path. 

    X_shortcut = X

    

    # strides: (1, 1) (for all layers in Identity block)

    kernel_initializer = he_normal(seed=0)

    

    # ### Main path ###

    

    # First component of main path

    X = Conv2D(filters=F1, kernel_size=(1, 1), padding='valid', name=f'{conv_name_base}2a', kernel_initializer=kernel_initializer)(X)

    X = BatchNormalization(name=f'{bn_name_base}2a')(X)

    X = Activation('relu')(X)

    

    # Second component of main path

    X = Conv2D(filters=F2, kernel_size=(f, f), padding='same', name=f'{conv_name_base}2b', kernel_initializer=kernel_initializer)(X)

    X = BatchNormalization(name=f'{bn_name_base}2b')(X)

    X = Activation('relu')(X)

    

    # Third component of main path

    X = Conv2D(filters=F3, kernel_size=(1, 1), padding='valid', name=f'{conv_name_base}2c', kernel_initializer=kernel_initializer)(X)

    X = BatchNormalization(name=f'{bn_name_base}2c')(X)

    

    # ### Shortcut path ###

    

    # Final step: Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)

    

    return X
def convolutional_block(X, f, filters, stage, block, s=2):

    """

    m -- number of examples

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    s -- Integer, specifying the stride to be used



    Returns:

    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    """

    

    # Defining name basis

    conv_name_base = f'res{stage}{block}_branch'

    bn_name_base = f'bn{stage}{block}_branch'

    

    # Retrieve filters

    F1, F2, F3 = filters

    

    # Save the input value. You'll need this later to add back to the main path. 

    X_shortcut = X

    

    # strides: (1, 1) (for all layers in Identity block)

    kernel_initializer = he_normal(seed=0)

    

    # ### Main path ###

    

    # First component of main path

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=f'{conv_name_base}2a', kernel_initializer=kernel_initializer)(X)

    X = BatchNormalization(name=f'{bn_name_base}2a')(X)

    X = Activation('relu')(X)

    

    # Second component of main path

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=f'{conv_name_base}2b', kernel_initializer=kernel_initializer)(X)

    X = BatchNormalization(name=f'{bn_name_base}2b')(X)

    X = Activation('relu')(X)

    

    # Third component of main path

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=f'{conv_name_base}2c', kernel_initializer=kernel_initializer)(X)

    X = BatchNormalization(name=f'{bn_name_base}2c')(X)

    

    # ### Shortcut path ###

    

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=f'{conv_name_base}1', kernel_initializer=kernel_initializer)(X_shortcut)

    X_shortcut = BatchNormalization(name=f'{bn_name_base}1')(X_shortcut)

    

    # Final step: Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)

    

    return X
def ResNet50(input_shape=(64, 64, 3), classes=1):

    """

    Implementation of the popular ResNet50 the following architecture:

    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3

    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER



    Arguments:

    input_shape -- shape of the images of the dataset

    classes -- integer, number of classes



    Returns:

    model -- a Model() instance in Tensorflow's Keras API

    """

    

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)

    

    # Zero-Padding

    X = ZeroPadding2D((3, 3))(X_input)

    

    # Stage 1

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=he_normal(seed=0))(X)

    X = BatchNormalization(name='bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    

    # Stage 2

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    

    # Stage 3

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    

    # Stage 4

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')



    # Stage 5 

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')



    # AVGPOOL

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)



    # output layer

    X = Flatten()(X)

    X = Dense(classes, activation='sigmoid', name=f'fc{classes}', kernel_initializer=he_normal(seed=0))(X)



    # Create model

    model = Model(inputs=X_input, outputs=X, name='ResNet50')



    return model