import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

import time

import os

import sys

from PIL import Image

import imageio

import glob

import math

from tqdm import tqdm

%matplotlib inline

print('tensorflow version:{}'.format(tf.__version__))
folder='../input/anime-faces/data/'

filenames=os.listdir(folder)
filepaths=[folder+filename for filename in filenames if filename.find('png')!=-1]

print('find images:{}'.format(len(filepaths)))
def get_dataset(filepaths):

    images=[]

    image_nums=len(filepaths)

    for i in tqdm(range(image_nums)):

        image=imageio.imread(filepaths[i])

        image=tf.cast(image,tf.float32)

        image=(image-127.5)/127.5

        images.append(image)

        #print('The image {} done'.format(i+1))

    #images=tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(64)

    print('find {} images for trianing'.format(image_nums))

    return images
images=get_dataset(filepaths)
def pre_show_images(images,show_nums):

    index=np.random.randint(0,len(images)-show_nums)

    figure=plt.figure(figsize=(10,10))

    row=math.sqrt(show_nums)

    col=row

    for i in range(show_nums):

        plt.subplot(row,col,i+1)

        plt.imshow((images[i+index]+1)/2)

        plt.axis('off')

    plt.subplots_adjust(wspace=0,hspace=0)

    plt.show()
pre_show_images(images,64)
batch_size=64

dataset=tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size)
def get_generator():

    model=keras.Sequential()

    

    model.add(keras.layers.Dense(8*8*256,input_shape=(100,),use_bias=False))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.LeakyReLU())

    

    model.add(keras.layers.Reshape((8,8,256)))

    assert model.output_shape==(None,8,8,256)

    

    model.add(keras.layers.Conv2DTranspose(128,5,1,use_bias=False,padding='same'))

    assert model.output_shape==(None,8,8,128)

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.LeakyReLU())

    

    model.add(keras.layers.Conv2DTranspose(64,5,2,use_bias=False,padding='same'))

    assert model.output_shape==(None,16,16,64)

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.LeakyReLU())

    

    model.add(keras.layers.Conv2DTranspose(32,5,2,use_bias=False,padding='same'))

    assert model.output_shape==(None,32,32,32)

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.LeakyReLU())

    

    model.add(keras.layers.Conv2DTranspose(3,5,2,use_bias=False,padding='same',activation='tanh'))

    assert model.output_shape==(None,64,64,3)

    return model
generator=get_generator()

generator.summary()
def get_discriminator():

    model=keras.Sequential()

    model.add(keras.layers.Conv2D(64,5,2,padding='same',input_shape=(64,64,3)))

    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Dropout(0.3))

    

    model.add(keras.layers.Conv2D(128,5,2,padding='same'))

    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Dropout(0.3))

    

    model.add(keras.layers.Conv2D(256,5,2,padding='same'))

    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Dropout(0.3))

    

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1))

    return model

    
discriminator=get_discriminator()

discriminator.summary()
cross_entropy=keras.losses.BinaryCrossentropy(from_logits=True)

def get_gen_loss(fake_out):

    return cross_entropy(tf.ones_like(fake_out),fake_out)

def get_dis_loss(real_out,fake_out):

    real_loss=cross_entropy(tf.ones_like(real_out),real_out)

    fake_loss=cross_entropy(tf.zeros_like(fake_out),fake_out)

    return real_loss+fake_loss

gen_opt=keras.optimizers.Adam(1e-4)

dis_opt=keras.optimizers.Adam(1e-4)
noise_dim=100

seed=tf.random.normal([batch_size,noise_dim]) #test for generator
@tf.function

def train_step(images):

    noise = tf.random.normal([batch_size, noise_dim])



    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:

        generated_images = generator(noise, training=True)



        real_output = discriminator(images, training=True)

        fake_output = discriminator(generated_images, training=True)



        gen_loss = get_gen_loss(fake_output)

        dis_loss = get_dis_loss(real_output, fake_output)



    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)

    dis_grad = dis_tape.gradient(dis_loss, discriminator.trainable_variables)



    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))

    dis_opt.apply_gradients(zip(dis_grad, discriminator.trainable_variables))
from tqdm import tqdm

def train(epochs, dataset):

    start = time.time()

    

    for epoch in tqdm(range(epochs)):

        for image_batch in dataset:

            train_step(image_batch)

    end=time.time()

    print ('Time for training is {} sec'.format(end-start))
%%time

epochs=2000

train(epochs,dataset)
def show_gen_images(generator,seed):

    show_nums=seed.shape[0]

    figure=plt.figure(figsize=(12,12))

    row=math.sqrt(show_nums)

    col=row

    images=generator(seed)

    for i in range(show_nums):

        plt.subplot(row,col,i+1)

        plt.imshow((images[i]+1)/2)

        plt.axis('off')

    plt.subplots_adjust(wspace=0,hspace=0)

    plt.show()
show_gen_images(generator,seed)
generator.save('generator.h5')