"""!pip install scipy==1.2.0

!pip install keras==2.1.6

!pip install tensorflow==1.7.0"""
import os

import sys

import numpy as np

from numpy import array

import matplotlib.pyplot as plt

import nibabel as nib

import cv2

from PIL import Image

from keras.layers import Dense

from keras.layers.core import Activation

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import UpSampling2D

from keras.layers.core import Flatten

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.models import Model

from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.layers import add

# Declare switch_backend will cause plots unable to display

plt.switch_backend('agg')

from keras.applications.vgg19 import VGG19

from keras.optimizers import SGD, Adam, RMSprop

import keras

import keras.backend as K

from keras.layers import Lambda, Input

import tensorflow as tf

import skimage.transform

from tqdm import tqdm

from skimage import data, io, filters

from skimage.transform import rescale, resize

import pandas as pd

import math

tf.config.run_functions_eagerly(True)

# from scipy.misc import imresize

# Code replace ment for imresize from scipy.misc

# np.array(Image.fromarray(arr).resize())
np.random.seed(10)

imshape = (256,256,3)

batch_size = 8
# Residual block



def res_block_gen(model, kernal_size, filters, strides):

    

    gen = model

    

    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)

    model = BatchNormalization(momentum = 0.5)(model)

    # Using Parametric ReLU

    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)

    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)

    model = BatchNormalization(momentum = 0.5)(model)

        

    model = add([gen, model])

    

    return model
# Up-sampling Block



def up_sampling_block(model, kernal_size, filters, strides):

    

    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)

    # Even we can have our own function for deconvolution (i.e one made in Utils.py)

    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)

    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)

    model = UpSampling2D(size = 2)(model)

    model = LeakyReLU(alpha = 0.2)(model)

    

    return model
# Discriminator Block



def discriminator_block(model, filters, kernel_size, strides):

    

    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)

    model = BatchNormalization(momentum = 0.5)(model)

    model = LeakyReLU(alpha = 0.2)(model)

    

    return model
class Generator(object):

    

    def __init__(self, noise_shape):     

        self.noise_shape = noise_shape

    

    def generator(self):

        

        gen_input = Input(shape = self.noise_shape)

    

        model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)

        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)

    

        gen_model = model

        

        # Using 16 Residual Blocks

        for index in range(16):

            model = res_block_gen(model, 3, 64, 1)

    

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)

        model = BatchNormalization(momentum = 0.5)(model)

        model = add([gen_model, model])

        

        # Using 2 UpSampling Blocks

        for index in range(2):

            model = up_sampling_block(model, 3, 256, 1)

    

        model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)

        model = Activation('tanh')(model)

   

        generator_model = Model(inputs = gen_input, outputs = model)

        

        return generator_model
class Discriminator(object):

    

    def __init__(self, image_shape):

        self.image_shape = image_shape

        

    def discriminator(self):

        dis_input = Input(shape = self.image_shape)

        

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)

        model = LeakyReLU(alpha = 0.2)(model)

        

        model = discriminator_block(model, 64, 3, 2)

        model = discriminator_block(model, 128, 3, 1)

        model = discriminator_block(model, 128, 3, 2)

        model = discriminator_block(model, 256, 3, 1)

        model = discriminator_block(model, 256, 3, 2)

        model = discriminator_block(model, 512, 3, 1)

        model = discriminator_block(model, 512, 3, 2)

        

        model = Flatten()(model)

        model = Dense(1024)(model)

        model = LeakyReLU(alpha = 0.2)(model)

       

        model = Dense(1)(model)

        model = Activation('sigmoid')(model) 

        

        discriminator_model = Model(inputs = dis_input, outputs = model)

        

        return discriminator_model
# Loss function

def vgg_loss(y_true, y_pred):

    

    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=imshape)

    vgg19.trainable = False

    for l in vgg19.layers:

        l.trainable = False

    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)

    loss_model.trainable = False

    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
# Create network

def get_gan_network(discriminator, shape, generator, optimizer):

    

    discriminator.trainable = False

    gan_input = Input(shape=shape)

    x = generator(gan_input)

    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=[x,gan_output])

    gan.compile(loss=[vgg_loss, "binary_crossentropy"],

                loss_weights=[1., 1e-3],

                optimizer=optimizer)



    return gan
# Normalize/Denormalize images

def normalize(input_data):

    return (input_data.astype(np.float32))/255



def denormalize(input_data):

    input_data = input_data * 255

    return input_data.astype(np.uint8)
# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)

def SubpixelConv2D(input_shape, scale=4):

    

    def subpixel_shape(input_shape):

        dims = [input_shape[0],input_shape[1] * scale,input_shape[2] * scale,int(input_shape[3] / (scale ** 2))]

        output_shape = tuple(dims)

        return output_shape

    

    def subpixel(x):

        return tf.depth_to_space(x, scale)

        

    return Lambda(subpixel, output_shape=subpixel_shape)
def load_data(path):

    files =[]

    for dirname, _, filenames in os.walk(path):

        for filename in filenames:

            files.append(cv2.imread(os.path.join(dirname, filename)))

    return files
f = load_data('../input/ohgodno/data')
def hr_images(images):

    images_hr = array(images)

    return images_hr



def lr_images(images_real , downscale):

    

    images = []

    for img in  range(len(images_real)):

        images.append(imresize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale], interp='bicubic', mode=None))

    images_lr = array(images)

    return images_lr
def plot_generated_images(epoch,generator, examples=3 , dim=(1, 3), figsize=(15, 5)):

    

    rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)

    image_batch_hr = denormalize(x_test_hr[rand_nums])

    image_batch_lr = x_test_lr[rand_nums]

    gen_img = generator.predict(image_batch_lr)

    generated_image = denormalize(gen_img)

    image_batch_lr = denormalize(image_batch_lr)

    

    #generated_image = deprocess_HR(generator.predict(image_batch_lr))

    

    plt.figure(figsize=figsize)

    

    plt.subplot(dim[0], dim[1], 1)

    plt.imshow(image_batch_lr[1], interpolation='nearest')

    plt.axis('off')

        

    plt.subplot(dim[0], dim[1], 2)

    plt.imshow(generated_image[1], interpolation='nearest')

    plt.axis('off')

    

    plt.subplot(dim[0], dim[1], 3)

    plt.imshow(image_batch_hr[1], interpolation='nearest')

    plt.axis('off')

    

    plt.tight_layout()

    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)
resize_shape = (64, 64)



x_train = f[:800]

x_test = f[800:1000]

    

x_train_hr = array(x_train)

x_train_hr = normalize(x_train_hr)

    

x_train_lr = [cv2.resize(i, resize_shape) for i in x_train]

x_train_lr = array(x_train_lr)

x_train_lr = normalize(x_train_lr)

    

x_test_hr = array(x_test)

x_test_hr = normalize(x_test_hr)

    

x_test_lr = [cv2.resize(i, resize_shape) for i in x_test]

x_test_lr = array(x_test_lr)

x_test_lr = normalize(x_test_lr)
batch_count = int(x_train_hr.shape[0] / batch_size)

shape = (64, 64, 3)



adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



generator = Generator(shape).generator()

discriminator = Discriminator(imshape).discriminator()



generator.compile(loss=vgg_loss, optimizer=adam)

discriminator.compile(loss="binary_crossentropy", optimizer=adam)



gan = get_gan_network(discriminator, shape, generator, adam)
loss_file = open('losses.txt' , 'w+')

loss_file.close()
epochs = 1

batch_count = int(x_train_hr.shape[0] / batch_size)

shape = (64, 64, 3)

    

generator = Generator(shape).generator()

discriminator = Discriminator(imshape).discriminator()



adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

generator.compile(loss=vgg_loss, optimizer=adam)

discriminator.compile(loss="binary_crossentropy", optimizer=adam)

    

shape = (64, 64, 3)

gan = get_gan_network(discriminator, shape, generator, adam)



best_loss = math.inf
for e in range(1, epochs+1):

    print ('-'*15, 'Epoch %d' % e, '-'*15)

    for _ in tqdm(range(batch_count)):

            

        rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)



        image_batch_hr = x_train_hr[rand_nums]

        image_batch_lr = x_train_lr[rand_nums]

        generated_images_sr = generator.predict(image_batch_lr)



        real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2

        fake_data_Y = np.random.random_sample(batch_size)*0.2

            

        discriminator.trainable = True



        d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)

        d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)

        discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            

        rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)

        image_batch_hr = x_train_hr[rand_nums]

        image_batch_lr = x_train_lr[rand_nums]



        gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2

        discriminator.trainable = False

        gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])

        tf.keras.backend.clear_session()

            

    print("Loss HR , Loss LR, Loss GAN")

    print(d_loss_real, d_loss_fake, gan_loss)

    loss_file = open('./losses.txt' , 'a')

    loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, discriminator_loss) )

    loss_file.close()

    if e == 1 or e % 5 == 0:

        plot_generated_images(e, generator)

    if gan_loss[2] < best_loss:

        generator.save('gen_model.h5')

        discriminator.save('dis_model.h5')

        gan.save('gan_model.h5')

        best_loss = gan_loss[2]