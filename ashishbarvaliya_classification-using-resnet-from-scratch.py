import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt



import tensorflow.keras

from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, AvgPool2D, Add, Dense 

from tensorflow.keras.preprocessing.image import load_img, img_to_array



import warnings

warnings.warn("ignore")

import os

os.chdir('../')

os.listdir('input')
from keras.preprocessing.image import load_img, img_to_array

import os



def convert(path,y):

    array=[]

    img_cat = []

    for img_path in os.listdir(path):

        img = load_img(path + img_path, target_size=(150,150))

        img = img_to_array(img)

        img = img/255.

        array.append(img)

        img_cat.append(y)

    return np.array(array), np.array(img_cat)
trainX_building, trainY_building  = convert("input/intel-image-classification/seg_train/seg_train/buildings/",0)

trainX_forest,trainY_forest  = convert("input/intel-image-classification/seg_train/seg_train/forest/",1)

trainX_glacier,trainY_glacier  = convert("input/intel-image-classification/seg_train/seg_train/glacier/",2)

trainX_mount,trainY_mount  = convert("input/intel-image-classification/seg_train/seg_train/mountain/",3)

trainX_sea,trainY_sea  = convert("input/intel-image-classification/seg_train/seg_train/sea/",4)

trainX_street,trainY_street  = convert("input/intel-image-classification/seg_train/seg_train/street/",5)



print('train building shape ', trainX_building.shape, trainY_building.shape) 

print('train forest', trainX_forest.shape ,trainY_forest.shape)

print('train glacier', trainX_glacier.shape,trainY_glacier.shape)

print('train mountain', trainX_mount.shape, trainY_mount.shape)

print('train sea',     trainX_sea.shape, trainY_sea.shape)

print('train street', trainX_street.shape ,trainY_street.shape)
X_train= np.concatenate((trainX_building,trainX_forest, trainX_glacier,trainX_mount, trainX_sea,trainX_street),axis=0)

y_train= np.concatenate((trainY_building,trainY_forest, trainY_glacier,trainY_mount, trainY_sea,trainY_street),axis=0)
testX_building, testY_building  = convert("input/intel-image-classification/seg_test/seg_test/buildings/",0)

testX_forest,testY_forest  = convert("input/intel-image-classification/seg_test/seg_test/forest/",1)

testX_glacier,testY_glacier  = convert("input/intel-image-classification/seg_test/seg_test/glacier/",2)

testX_mount,testY_mount  = convert("input/intel-image-classification/seg_test/seg_test/mountain/",3)

testX_sea,testY_sea  = convert("input/intel-image-classification/seg_test/seg_test/sea/",4)

testX_street,testY_street  = convert("input/intel-image-classification/seg_test/seg_test/street/",5)



print('test building shape ', testX_building.shape, testY_building.shape) 

print('test forest', testX_forest.shape ,testY_forest.shape)

print('test glacier', testX_glacier.shape,testY_glacier.shape)

print('test mountain', testX_mount.shape, testY_mount.shape)

print('test sea',     testX_sea.shape, testY_sea.shape)

print('test street', testX_street.shape ,testY_street.shape)
X_test= np.concatenate((testX_building,testX_forest, testX_glacier,testX_mount, testX_sea,testX_street),axis=0)

y_test= np.concatenate((testY_building,testY_forest, testY_glacier,testY_mount, testY_sea,testY_street),axis=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from tensorflow.keras.utils import to_categorical



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

y_train.shape,y_test.shape

y_test[1]
from tensorflow.keras.layers import Input,Dropout, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D

from tensorflow.keras.models import Model, load_model,Sequential

from tensorflow.keras.initializers import glorot_uniform
def identity_block(X, f, filters, stage, block):

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'



    # Retrieve Filters

    F1, F2, F3 = filters



    # Save the input value. We'll need this later to add back to the main path. 

    X_shortcut = X



    # First component of main path

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)



    # Second component of main path

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)



    return X
def convolutional_block(X, f, filters, stage, block, s = 2):

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value

    X_shortcut = X





    ##### MAIN PATH #####

    # First component of main path 

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)



    # Second component of main path

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)



    

    ##### SHORTCUT PATH ####

    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)

    

    return X


def ResNet50(input_shape = (150, 150, 3), classes = 6):   

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    # Zero-Padding

    X = ZeroPadding2D((3, 3))(X_input)

    

    # Stage 1

    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)



    # Stage 2

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')



    # Stage 3

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')

    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')



    # Stage 4

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')



    # Stage 5

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')



    # AVGPOOL.

    X = AveragePooling2D((2, 2), name='avg_pool')(X)



    # output layer

    X = Flatten()(X)

    X = Dropout(0.3)(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    

    # Create model

    model = Model(inputs = X_input, outputs = X, name='ResNet50')



    return model
model = ResNet50(input_shape = (150, 150, 3), classes = 6)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs = 50, batch_size = 64,validation_data=(X_test, y_test), shuffle=True)
model.save('/my_model.h5')

model.evaluate(X_test, y_test)