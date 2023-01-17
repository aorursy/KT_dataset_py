import glob
import cv2
import os
import numpy as np

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Activation, Flatten,Reshape,AveragePooling2D,ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization,Add,Dropout,Flatten,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, concatenate
from keras.utils import plot_model
def UNet_latent_custom():
    input_size = (288, 480, 4)
    inp = Input(input_size) 
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    latent_entrance = Flatten()(conv5)
    latent_layer = Dense(256, activation="elu")(latent_entrance)
    z_mean = Dense(64, name="z_mean")(latent_layer)
    z_log_var = Dense(64, name="z_log_var")(latent_layer)
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    z_sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    z_output = Dense(144 *240*128, activation="relu")(z_sample)
    z_output = Reshape((144, 240, 128))(z_output)
    
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    c
    
    
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8,z_output], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9,inp[:,:,:,3:4]], axis=3)
    conv9 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    coarse_decoder_model = Model(inputs=inp, outputs=[conv10,z_sample])
    return coarse_decoder_model 
aaa = UNet_latent_custom()
aaa.summary()
plot_model(aaa)
def UNet_latent_light_custom():
    input_size = (288, 480, 3)
    inp = Input(input_size) 
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    
    random_noise = tf.keras.backend.random_normal(shape=(36,60,512))
    z_mean = Dense(36*60,activation='elu', name="z_mean")(Flatten()(conv4[:,:,:,0:1]))
    
    z_mean = Reshape((36, 60,512))(z_mean)
    
    latent_layer = z_mean+tf.exp(0.5 * conv4)*(random_noise)
    latent_layer = Conv2D(512, 2, activation='elu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(latent_layer))
    
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop4))
    
    merge7 = concatenate([UpSampling2D(size=(2, 2))(conv4), up6,latent_layer], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9,inp[:,:,:,3:4]], axis=3)
    conv9 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    coarse_decoder_model = Model(inputs=inp, outputs=[conv10])
    return coarse_decoder_model 
bbb = UNet_latent_light_custom()
bbb.summary()
plot_model(bbb)
def UNet_inp_custom():
    input_size = (288, 480, 5)
    inp = Input(input_size) 
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9,inp[:,:,:,3:4]], axis=3)
    conv9 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    coarse_decoder_model = Model(inputs=inp, outputs=[conv10])
    return coarse_decoder_model 
bbb =UNet_inp_custom()
plot_model(bbb)
def UNet_inp_custom():
    input_size = (288,4800 , 5)
    inp = Input(input_size) 
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9,inp[:,:,:,3:4]], axis=3)
    conv9 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    coarse_decoder_model = Model(inputs=inp, outputs=[conv10])
    return coarse_decoder_model 
aaa =UNet_inp_custom()
plot_model(aaa)
import keras.backend as K
def refinement_model():
    input_size = (288, 480, 4)
    inp = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    before_shape = (K.int_shape(pool2))
    latent = Flatten()(pool2)
    latent_dense = Dense(64,activation='elu')(latent)
    latent_dense = Reshape(before_shape[1:])(latent_dense)
    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(latent_dense))
    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(up6))
    up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(up7))
    up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(up8))
    
    res = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(up9)
    
    
    coarse_decoder_model = Model(inputs=inp, outputs=[res])
    return coarse_decoder_model
sss =refinement_model()
sss.summary()
def simple_unet_finement_model():
    input_size = (288, 480, 4)
    inp = Input(input_size)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(pool2))
    merge6 = concatenate([UpSampling2D(size=(2, 2))(pool2), up8], axis=3)
    
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))((merge6)))
    merge7 = concatenate([UpSampling2D(size=(2, 2))(conv2), up9], axis=3)
    up10 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))((merge7)))
    merge8 = concatenate([UpSampling2D(size=(2, 2))(conv1), up10], axis=3)
    
    conv10 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv_out = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv_out = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv_out)
    coarse_decoder_model = Model(inputs=inp, outputs=[conv_out])
    return coarse_decoder_model
aaa =simplerefinement_model()
aaa.summary()
plot_model(aaa)
def PSP_segmentation_custom():
    input_size = (288, 480, 3)
    inp = Input(input_size)
    stmp_conv3x3 = Conv2D(32, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    stmp_conv3x3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(stmp_conv3x3)
    stmp_conv3x3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(stmp_conv3x3)
    stmp_conv3x3_mx = ZeroPadding2D(padding=(1, 1))(stmp_conv3x3)
    stmp_conv3x3_mx = Conv2D(96, 1, activation='relu', padding='same', kernel_initializer='he_normal')(stmp_conv3x3_mx)
    stmp_conv3x3_96 = Conv2D(96, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        stmp_conv3x3)
    stmp_pool = pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(stmp_conv3x3_mx)
    stmp_merge = concatenate([stmp_pool, stmp_conv3x3_96], axis=3)

    conv1x1 = Conv2D(96, 1, activation='relu', padding='same', kernel_initializer='he_normal')(stmp_merge)

    conv3x3 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(stmp_merge)
    conv3x3 = Conv2D(96, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3x3)

    conv5x5 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(stmp_merge)
    conv5x5 = Conv2D(96, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv5x5)

    conv7x7 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(stmp_merge)
    conv7x7 = Conv2D(96, 7, activation='relu', padding='same', kernel_initializer='he_normal')(conv7x7)

    iv4module_1 = concatenate([conv1x1, conv3x3, conv5x5, conv7x7], axis=3)

    conv1x1 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(iv4module_1)
    

    conv3x3 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(iv4module_1)
    conv3x3 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3x3)

    conv5x5 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(iv4module_1)
    conv5x5 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv5x5)

    conv7x7 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(iv4module_1)
    conv7x7 = Conv2D(256, 7, activation='relu', padding='same', kernel_initializer='he_normal')(conv7x7)
    
    
    iv4module_2 = concatenate([conv1x1, conv3x3, conv5x5, conv7x7], axis=3)
    
    
    conv1x1_iv2 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(iv4module_2)
    conv1x1_iv2_pool = pool2 = MaxPooling2D(pool_size=(2, 2))(conv1x1_iv2)
    
    conv3x3_iv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(iv4module_2)
    conv3x3_iv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv3x3_iv2)
    conv3x3_iv2 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3x3_iv2)
    conv3x3_iv2_pool = pool2 = MaxPooling2D(pool_size=(2, 2))(conv3x3_iv2)

    conv5x5_iv2 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(iv4module_2)
    conv5x5_iv2_pool = pool2 = MaxPooling2D(pool_size=(2, 2))(conv5x5_iv2)
    
    conv7x7_iv2 = Conv2D(64, 1, padding='same', kernel_initializer='he_normal')(iv4module_2)
    conv7x7_iv2 = Conv2D(128, 7, padding='same', kernel_initializer='he_normal')(conv7x7_iv2)
    conv7x7_iv2 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv7x7_iv2)
    conv7x7_iv2_pool = pool2 = MaxPooling2D(pool_size=(2, 2))(conv7x7_iv2)

    iv4module_3 = concatenate([conv1x1_iv2_pool, conv5x5_iv2_pool,conv3x3_iv2_pool, conv7x7_iv2_pool], axis=3)
    
    conv1x1_iv3 = Conv2D(1024, 1, padding='same', kernel_initializer='he_normal')(iv4module_3)
    conv1x1_iv3 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(conv1x1_iv3)
    conv1x1_iv3_pool = pool2 = MaxPooling2D(pool_size=(2, 2))(conv1x1_iv3)
    
    conv3x3_iv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(iv4module_3)
    conv3x3_iv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv3x3_iv3)
    conv3x3_iv3 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3x3_iv3)
    conv3x3_iv3_pool = pool2 = MaxPooling2D(pool_size=(2, 2))(conv3x3_iv3)
    
    conv5x5_iv3 = Conv2D(64, 1, padding='same', kernel_initializer='he_normal')(iv4module_3)
    conv5x5_iv3 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv5x5_iv3)
    conv5x5_iv3_pool = pool2 = MaxPooling2D(pool_size=(2, 2))(conv5x5_iv3)
    
    conv7x7_iv3 = Conv2D(64, 1, padding='same', kernel_initializer='he_normal')(iv4module_3)
    conv7x7_iv3 = Conv2D(128, 7, padding='same', kernel_initializer='he_normal')(conv7x7_iv3)
    conv7x7_iv3 = Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(conv7x7_iv3)
    conv7x7_iv3_pool = pool2 = MaxPooling2D(pool_size=(2, 2))(conv7x7_iv3)
    
    iv4module_4 = concatenate([conv1x1_iv3_pool, conv3x3_iv3_pool, conv5x5_iv3_pool, conv7x7_iv3_pool], axis=3)
    
    
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(iv4module_4))
    
    merge7 = concatenate([UpSampling2D(size=(2, 2))(iv4module_4), up6], axis=3)
    
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(iv4module_3))
    
    merge8 = concatenate([UpSampling2D(size=(2, 2))(conv7),up7], axis=3)
    
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    up8 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(iv4module_2))
    
    merge9 = concatenate([UpSampling2D(size=(2, 2))(conv8), up8], axis=3)
    
    conv9 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    up9 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(iv4module_1))
    
    merge10 = concatenate([conv9, up9,UpSampling2D(size=(2, 2))(stmp_merge)], axis=3)
    
    conv10 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv10 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    
    up10 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv10))
    
    conv11 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(up10)
    conv11 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv11)
    model = Model(inputs=inp, outputs=[conv11])
    
    return model
    
rrr =PSP_segmentation_custom()
rrr.summary()
plot_model(rrr)

