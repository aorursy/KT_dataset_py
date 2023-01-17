from __future__ import print_function, division

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import os

import scipy

from glob import glob

import numpy as np

import matplotlib.pyplot as plt

from skimage import transform

from PIL import Image  

import datetime

from skimage import transform

import sys

from imageio import imread
xpaths = []

ypaths = []

path = '../input/kaist-dataset'

sets = os.listdir(path)

sets.remove('set05')



for s in sets:

    set_path = path + '/' + s

    V = os.listdir(set_path)

    for v in V:

        ir = set_path + '/' + v + '/lwir'

        for image in os.listdir(ir):

            xpaths.append(ir + '/' + image)

        

        vis = set_path + '/' + v + '/visible'

        for image in os.listdir(vis):

            ypaths.append(vis + '/' + image)

        print(len(xpaths))    



for image in os.listdir(path + '/set05/lwir'):

    xpaths.append(path + '/set05/lwir/' + image)



for image in os.listdir(path + '/set05/visible'):

    ypaths.append(path + '/set05/visible/' + image)

    
image = plt.imread(ypaths[47264])

plt.imshow(image)
def create_data(dataset):

    xpaths = []

    ypaths = []

    if dataset == "cityscapes":

        path = '../input/cityscapes-dataset/cityscapes/train'

        for filename in os.listdir(path):

            xpaths.append(path+'/'+filename)

        return xpaths,ypaths

    elif dataset == 'facades':

        path = '../input/facade-dataset/base'

        for filename in sorted(os.listdir(path)):

            if '.png' in filename:

                xpaths.append(path+'/'+filename)

            elif '.jpg' in filename:

                ypaths.append(path+'/'+filename)

        return xpaths,ypaths    

    else:

        path = '../input/maps-dataset/maps/train'

        for filename in os.listdir(path):

            xpaths.append(path+'/'+filename)

        return xpaths,ypaths

#xpaths,ypaths =  create_data('maps')
#xpaths,ypaths =  create_data('cityscapes')
len(xpaths)
from keras.models import Model

from keras import layers

from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape

from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D, Conv2DTranspose

from keras.layers import Input, Concatenate,Add

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.normalization import BatchNormalization

from keras.initializers import RandomNormal

from keras.layers.pooling import MaxPooling2D,AveragePooling2D

import keras.backend as K

import numpy as np

import keras

from keras.optimizers import Adam

from keras.utils import plot_model



!pip install git+https://www.github.com/keras-team/keras-contrib.git
def define_encoder_block_unet(layer_in, n_filters, batchnorm=True):

    # weight initialization

    init = RandomNormal(stddev=0.02) 

    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)

    if batchnorm:

         g = BatchNormalization()(g, training=True)

    g = LeakyReLU(alpha=0.2)(g)

    return g
def define_decoder_block_unet(layer_in, skip_in, n_filters, dropout=True):

    # weight initialization

    init = RandomNormal(stddev=0.02)

    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)

    g = BatchNormalization()(g, training=True)

    if dropout:

         g = Dropout(0.5)(g, training=True)

    # merge with skip connection

    g = Concatenate()([g, skip_in])

    g = Activation('relu')(g)

    return g
def unet(image_shape):

    # weight initialization

    init = RandomNormal(stddev=0.02)

    # image input

    in_image = Input(shape=image_shape)

    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512

    e1 = define_encoder_block_unet(in_image, 64, batchnorm=False)

    e2 = define_encoder_block_unet(e1, 128)

    e3 = define_encoder_block_unet(e2, 256)

    e4 = define_encoder_block_unet(e3, 512)

    e5 = define_encoder_block_unet(e4, 512)

    e6 = define_encoder_block_unet(e5, 512)

    e7 = define_encoder_block_unet(e6, 512)

    # bottleneck, no batch norm and relu

    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)

    b = Activation('relu')(b)

    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

    d1 = define_decoder_block_unet(b, e7, 512)

    d2 = define_decoder_block_unet(d1, e6, 512)

    d3 = define_decoder_block_unet(d2, e5, 512)

    d4 = define_decoder_block_unet(d3, e4, 512, dropout=False)

    d5 = define_decoder_block_unet(d4, e3, 256, dropout=False)

    d6 = define_decoder_block_unet(d5, e2, 128, dropout=False)

    d7 = define_decoder_block_unet(d6, e1, 64, dropout=False)

    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)

    out_image = Activation('tanh')(g)

    # define model

    model = Model(in_image, out_image)

    return model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
def define_encoder_block(layer_in, n_filters, instancenorm = True, change = False, no_strides = False, residual_block = False):

    # weight initialization

    init = RandomNormal(stddev=0.02)

    # add downsampling layer

    if no_strides :

        stride = (1,1)

    else:

        stride = (2,2)

    if(change):

        print('in seven')

        g = Conv2D(n_filters, (7,7), strides=stride, padding='same', kernel_initializer=init)(layer_in)

    else:

        g = Conv2D(n_filters, (3,3), strides=stride, padding='same', kernel_initializer=init)(layer_in)

    # conditionally add batch normalization

    if instancenorm:

        g = InstanceNormalization()(g, training=True)

    # leaky relu activation

    g = LeakyReLU(alpha=0.2)(g)

    

    # if this is a residual block

    if residual_block:

        g = Conv2D(n_filters, (3,3), strides=stride, padding='same', kernel_initializer=init)(g)

        g = InstanceNormalization()(g, training=True)

        g = Add()([g, layer_in])

    

    return g
def define_decoder_block(layer_in,  n_filters):

    # weight initialization

    init = RandomNormal(stddev=0.02)

    # add upsampling layer

    g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)

    # add batch normalization

    g = InstanceNormalization()(g, training=True)

    # conditionally add dropout

    g = Activation('relu')(g)

    return g

def define_discriminator(image_shape):

    # weight initialization

    init = RandomNormal(stddev=0.02)

    # source image input

    in_src_image = Input(shape=image_shape)

    # target image input

    in_target_image = Input(shape=image_shape)

    # concatenate images channel-wise

    merged = Concatenate()([in_src_image, in_target_image])

    # C64

    d = Conv2D(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)

    d = LeakyReLU(alpha=0.2)(d)

    # C128

    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)

    d = InstanceNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    # C256

    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)

    d = InstanceNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    # C512

    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)

    d = InstanceNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    

    # patch output

    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)

    patch_out = Activation('sigmoid',name = 'dis_out')(d)

    # define model

    model = Model([in_src_image, in_target_image], patch_out)

    # compile model

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model

def chinese(image_shape):

    image_shape = (256,256,3)

    in_image = Input(shape=image_shape)

    g12 = define_encoder_block(in_image,64,change = True)

    print("g12 ",g12.shape)

    g13 = define_encoder_block(g12,64)

    print("g13 ",g13.shape)

    g14 = define_encoder_block(g13,128)

    print("g14 ",g14.shape)

    g15 = define_encoder_block(g14,256)

    print("g15 ",g15.shape)



    #block1

    out_1 = define_encoder_block(g15,256,no_strides = True, residual_block = True)



    #block2

    out_2 = define_encoder_block(out_1,256,no_strides = True, residual_block = True)



    #block3

    out_3 = define_encoder_block(out_2,256,no_strides = True, residual_block = True)



    #block4

    out_4 = define_encoder_block(out_3,256,no_strides = True, residual_block = True)



    #block5

    out_5 = define_encoder_block(out_4,256,no_strides = True, residual_block = True)



    #block6

    out_6 = define_encoder_block(out_5,256,no_strides = True, residual_block = True)



    #block7

    out_7 = define_encoder_block(out_6,256,no_strides = True, residual_block = True)



    #block8

    out_8 = define_encoder_block(out_7,256,no_strides = True, residual_block = True)



    #block9

    out_9 = define_encoder_block(out_8,256,no_strides = True, residual_block = True)



    g16 = define_decoder_block(out_9,256)

    print("g16 ",g16.shape)

    g17 = define_decoder_block(g16,128)

    print("g17 ",g17.shape)

    g18 = define_decoder_block(g17,64) 

    print("g18 ",g18.shape)

    g21 = define_encoder_block(in_image,32,change = True,no_strides = True)

    print(g21.shape)

    g22 = define_encoder_block(g21,64)

    print(g22.shape)

    merged = Add()([g22, g18])

    print("merged: ",merged.shape)



    #block1

    g31 = define_encoder_block(merged,256,no_strides = True)

    g32 = define_encoder_block(g31,256,no_strides = True)

    print("b1",g32.shape)



    #block2

    g33 = define_encoder_block(g32,256,no_strides = True)

    g34 = define_encoder_block(g33,256,no_strides = True)

    print("b2",g34.shape)



    #block3

    merged = Add()([g32,g34])

    g35 = define_encoder_block(merged,256,no_strides = True)

    g36 = define_encoder_block(g35,256,no_strides = True)

    print("b3",g36.shape)





    g37 = Add()([merged,g36])    

    g38 = define_decoder_block(g37,32)

    print("decoder",g38.shape)

    g39 = define_encoder_block(g38,3,no_strides = True)





    output = Activation('tanh',name = 'gen_out')(g39) 

    g_model = Model(in_image,output)

    return g_model





def generator(model,image_shape):

    if model == 'unet':

            return unet(image_shape)

    else:

            return chinese(image_shape)

            
from keras.applications.vgg16 import VGG16
def vgg(shape):

    image_shape = shape

    img_input = Input(shape=image_shape)



    #block1

    x = Conv2D(64, (3, 3),

                          padding='same',

                          name='block1_conv1')(img_input)

    x = Activation('relu',name = 'relu1_1')(x)

    x = Conv2D(64, (3, 3),

                          padding='same',

                          name='block1_conv2')(x)

    x = Activation('relu',name = 'relu1_2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)



    #block2

    x = layers.Conv2D(128, (3, 3),

                          padding='same',

                          name='block2_conv1')(x)

    x = Activation('relu',name = 'relu2_1')(x)

    x = layers.Conv2D(128, (3, 3),

                      padding='same',

                      name='block2_conv2')(x)

    x = Activation('relu',name = 'relu2_2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)





    #block3

    x = layers.Conv2D(256, (3, 3),

                          padding='same',

                          name='block3_conv1')(x)

    x = Activation('relu',name = 'relu3_1')(x)

    x = layers.Conv2D(256, (3, 3),

                      padding='same',

                      name='block3_conv2')(x)

    x = Activation('relu',name = 'relu3_2')(x)

    x = layers.Conv2D(256, (3, 3),

                      padding='same',

                      name='block3_conv3')(x)

    x = Activation('relu',name = 'relu3_3')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)





    #block4

    x = layers.Conv2D(512, (3, 3),

                          padding='same',

                          name='block4_conv1')(x)

    x = Activation('relu',name = 'relu4_1')(x)

    x = layers.Conv2D(512, (3, 3),

                       padding='same',

                      name='block4_conv2')(x)

    x = Activation('relu',name = 'relu4_2')(x)

    x = layers.Conv2D(512, (3, 3),

                        padding='same',

                      name='block4_conv3')(x)

    x = Activation('relu',name = 'relu4_3')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)





    #block5

    x = layers.Conv2D(512, (3, 3),

                           padding='same',

                          name='block5_conv1')(x)

    x = Activation('relu',name = 'relu5_1')(x)

    x = layers.Conv2D(512, (3, 3),

                       padding='same',

                      name='block5_conv2')(x)

    x = Activation('relu',name = 'relu5_2')(x)

    x = layers.Conv2D(512, (3, 3),

                      padding='same',

                      name='block5_conv3')(x)

    x = Activation('relu',name = 'relu5_3')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    vgg_model = Model(img_input,x)

    vgg_model.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    vgg_model.trainable=False

    for layer in vgg_model.layers:

        layer.trainable=False

    

    return vgg_model
from keras.applications.vgg16 import VGG16 as PTModel

def define_gan(g_model, d_model, image_shape):

    selectedLayers = ['relu1_1','relu2_1','relu3_2','relu4_2']

    vgg_model = vgg(image_shape)

    vgg_model.trainable = False

    vgg_model.summary()

    selectedOutputs = [vgg_model.get_layer(i).output for i in selectedLayers]

    #print(selectedOutputs)    

    vgg_model = Model(vgg_model.inputs,selectedOutputs,name ='lossModel')

    vgg_model.trainable = False

    for layer in vgg_model.layers:

        layer.trainable = False

    in_src = Input(shape=image_shape)

    gen_out = g_model(in_src)

    lossModelOutputs = vgg_model(gen_out)

    #print(lossModelOutputs)

    d_model.trainable = False

    dis_out = d_model([in_src, gen_out])

    outputs= [dis_out, gen_out] + lossModelOutputs

    print(outputs)

    model = Model(in_src, outputs)

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss=['binary_crossentropy','mae','mse','mse','mse','mse'], optimizer=opt, loss_weights=[1,100,1,1,1,1])

    print(model.summary())

    plot_model(model, to_file='model.png')

    return model,vgg_model
image_shape = (256,256,3)

d_model = define_discriminator(image_shape)

g_model = generator('unet',image_shape)

#g_model = generator('chinese',image_shape)

gan_model,loss_model = define_gan(g_model, d_model, image_shape)

gan_model.summary()

print(gan_model.output_shape)

print(loss_model.summary())

 

# generate a batch of images, returns images and targets

def generate_fake_samples(g_model, samples, n_patch):

    # generate fake instance

    X = g_model.predict(samples)

    # create 'fake' class labels (0)

    y = np.zeros((len(X), n_patch, n_patch, 1))

    return (X, y)
from random import randint
def show_images(x,y,z):

    plt.axis('off')

    fig = plt.figure(figsize = (21,7))

    plt.subplot(1,3,1)

    plt.imshow(x)

    plt.subplot(1,3,2)

    plt.imshow(y)

    plt.subplot(1,3,3)

    plt.imshow(z)

    plt.show()
import random

import matplotlib.pyplot as plt

import skimage.transform



# train pix2pix models

def train_cityscapes(d_model, g_model, gan_model,loss_model,shape, n_epochs=2, n_batch=1, n_patch=32):

    

    # calculate the number of batches per training epoch

    bat_per_epo = int(len(xpaths)/n_batch)

    

    # calculate the number of training iterations

    n_steps = bat_per_epo * n_epochs

    

    # index to iterate through path lists

    index = 0

    X_realA = np.ndarray(shape=(n_batch,shape[0],shape[1],3),dtype='float32')

    X_realB = np.ndarray(shape=(n_batch,shape[0],shape[1],3),dtype='float32')



    # manually enumerate epochs

    for i in range(n_steps):

        

        # select a batch of real samples

        for count in range(n_batch):

            image = plt.imread(xpaths[index])

            ximage = image[:,:600,:]

            yimage = image[:,600:,:]

            newximage = skimage.transform.resize(ximage, (shape[0], shape[1]), mode='constant')

            newyimage = skimage.transform.resize(yimage, (shape[0], shape[1]), mode='constant')

            X_realA[count] = newximage

            X_realB[count] = newyimage

            index = (index+1)%len(xpaths)

        y_real = np.ones((n_batch, n_patch, n_patch, 1))

         

        # generate a batch of fake samples

        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        d_model.trainable = True

        # update discriminator for real samples

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)

        

        # update discriminator for generated samples

        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        Y_train_lossModel = loss_model.predict_on_batch(X_realB)

        # print(Y_train_lossModel)

        # print(Y_train_lossModel[0].shape)

        # update the generator

        s1 = loss_model.get_layer('relu1_1').output_shape

        s2 = loss_model.get_layer('relu2_1').output_shape

        s3 = loss_model.get_layer('relu3_2').output_shape

        s4 = loss_model.get_layer('relu4_2').output_shape

        y1 = np.ndarray(shape=(n_batch,s1[1],s1[2],s1[3]),dtype='float32')

        y2 = np.ndarray(shape=(n_batch,s2[1],s2[2],s2[3]),dtype='float32')

        y3 = np.ndarray(shape=(n_batch,s3[1],s3[2],s3[3]),dtype='float32')

        y4 = np.ndarray(shape=(n_batch,s4[1],s4[2],s4[3]),dtype='float32')

        y1 = Y_train_lossModel[0]

        y2 = Y_train_lossModel[1]

        y3 = Y_train_lossModel[2]

        y4 = Y_train_lossModel[3]

        #g_loss, _, _

        d_model.trainable = False

        history = gan_model.fit(X_realA, [y_real, X_realB, y1, y2, y3, y4])

        ans = gan_model.predict(X_realA)

        show_images(X_realB[0],ans[1][0],X_realA[0])

        if (i+1)%100 == 0:

            outs = 'echo Step %d of %d: Loss d1[%.3f] d2[%.3f]' % (i+1, n_steps, d_loss1, d_loss2)

            print(outs)

            os.system(outs)

       

       
import random

import matplotlib.pyplot as plt

import skimage.transform



# train pix2pix models

def train_facades(d_model, g_model, gan_model,loss_model,shape, n_epochs=2, n_batch=1, n_patch=16):

    

    # calculate the number of batches per training epoch

    bat_per_epo = int(len(xpaths)/n_batch)

    

    # calculate the number of training iterations

    n_steps = bat_per_epo * n_epochs

    

    # index to iterate through path lists

    index = 0

    

    # manually enumerate epochs

    for i in range(n_steps):

        

        # select a batch of real samples

        X_realA = np.ndarray(shape=(n_batch,shape[0],shape[1],3),dtype='float32')

        X_realB = np.ndarray(shape=(n_batch,shape[0],shape[1],3),dtype='float32')



        for count in range(n_batch):

            ximage = plt.imread(xpaths[index])

            yimage = plt.imread(ypaths[index])

            newximage = skimage.transform.resize(ximage, (shape[0], shape[1]), mode='constant')

            newyimage = skimage.transform.resize(yimage, (shape[0], shape[1]), mode='constant')

            X_realA[count] = newximage

            X_realB[count] = newyimage

            index = (index+1)%len(xpaths)

        y_real = np.ones((n_batch, n_patch, n_patch, 1))

         # generate a batch of fake samples

        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        d_model.trainable = True

        # update discriminator for real samples

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)

        

        # update discriminator for generated samples

        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        Y_train_lossModel = loss_model.predict_on_batch(X_realB)

        # print(Y_train_lossModel)

        # print(Y_train_lossModel[0].shape)

        # update the generator

        s1 = loss_model.get_layer('relu1_1').output_shape

        s2 = loss_model.get_layer('relu2_1').output_shape

        s3 = loss_model.get_layer('relu3_2').output_shape

        s4 = loss_model.get_layer('relu4_2').output_shape

        y1 = np.ndarray(shape=(n_batch,s1[1],s1[2],s1[3]),dtype='float32')

        y2 = np.ndarray(shape=(n_batch,s2[1],s2[2],s2[3]),dtype='float32')

        y3 = np.ndarray(shape=(n_batch,s3[1],s3[2],s3[3]),dtype='float32')

        y4 = np.ndarray(shape=(n_batch,s4[1],s4[2],s4[3]),dtype='float32')

        y1 = Y_train_lossModel[0]

        y2 = Y_train_lossModel[1]

        y3 = Y_train_lossModel[2]

        y4 = Y_train_lossModel[3]

        #g_loss, _, _

        d_model.trainable = False

        history = gan_model.fit(X_realA, [y_real,  X_realB, y1, y2, y3, y4],verbose = 0)

        ans = gan_model.predict(X_realA)

        #show_images(X_realB[0],ans[1][0],X_realA[0])

        if i<50000 and i%100 == 0:

            show_images(X_realB[0],ans[1][0],X_realA[0])

        if i>50000 and i%200 == 0:

            show_images(X_realB[0],ans[1][0],X_realA[0])

        if (i+1)%100 == 0:

            outs = 'echo Step %d of %d: Loss d1[%.3f] d2[%.3f]' % (i+1, n_steps, d_loss1, d_loss2)

            os.system(outs)

        
train_facades(d_model,g_model,gan_model,loss_model,image_shape)