import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.misc import imread



import os

from IPython import display

import PIL

import time



import glob

#import imageio



import warnings

warnings.filterwarnings(action='ignore')

# Any results you write to the current directory are saved as output.



import gc



import tensorflow as tf

tf.enable_eager_execution()

train= pd.read_csv('/kaggle/input/mnist_train.csv')

test= pd.read_csv('/kaggle/input/mnist_test.csv')

train = pd.concat([train, test])

train.drop(train.columns[0], axis=1, inplace=True)

train = np.array(train)

train = train.reshape(train.shape[0],28,28,1).astype('float32')

train = (train -127.5) / 127.5

del test

gc.collect()

print(train.shape)

plt.imshow(train[0].reshape(28,28), cmap='gray')

plt.show()
BUFFER_S = 70000

BATCH_S = 256

# Create batches and shuffle the dataset

traindata = tf.data.Dataset.from_tensor_slices(train).shuffle(BUFFER_S).batch(BATCH_S)
def Generator():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,))) # input the random noise to the dense layer 

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7,7,256)))

    

    assert model.output_shape == (None,7,7,256)

    model.add(tf.keras.layers.Convolution2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False)) #[[outshape = (n-f+2p)/s + 1]]

    assert model.output_shape ==(None,7,7,128)

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Convolution2DTranspose(64,(5,5), strides=(2,2), padding='same', use_bias=False))

    assert model.output_shape == (None, 14,14,64)

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    

    model.add(tf.keras.layers.Convolution2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False))

    assert model.output_shape == (None, 28,28,1)

    

    return model

    
def Discriminator():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same'))

    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dropout(0.3))

    

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dropout(0.3))

    

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1))

    

    return model
# instantiate the models

generator = Generator()

discriminator = Discriminator()


def gLoss(generated_output):

    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)
def dLoss(real_output, generated_output):

    real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_output), real_output)

    gen_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(generated_output), generated_output)

    

    total_loss = real_loss + gen_loss

    return total_loss

    
gOptimizer = tf.train.AdamOptimizer(1e-4)

dOptimizer = tf.train.AdamOptimizer(1e-4)
chkpt_dir = './trng_chkpts'

chkpt_prefix = os.path.join(chkpt_dir,'ckpt')

checkpt = tf.train.Checkpoint(generator=generator, discriminator=discriminator, generator_optimizer=gOptimizer, discriminator_optimizer=dOptimizer)
noise_dim = 100

EPOCHS = 50

results = 10

random_vector = tf.random_normal([results, noise_dim])
def gantraining(images):

    noise = tf.random_normal([BATCH_S, noise_dim])

    

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        

        #Outputs from models

        real_output = discriminator(images, training=True)

        generated_output = discriminator(generated_images, training=True)

        

        #respective losses

        gen_loss = gLoss(generated_output)

        disc_loss = dLoss(real_output, generated_output)

    

    #calculate gradients

    grad_gen = gen_tape.gradient(gen_loss, generator.variables)

    grad_disc = disc_tape.gradient(disc_loss, discriminator.variables)

    

    #Apply gradients from optimizer

    gOptimizer.apply_gradients(zip(grad_gen,generator.variables))

    dOptimizer.apply_gradients(zip(grad_disc, discriminator.variables))
def image_processing(model, epoch, test_input):

    #predictions at each epoch using the input random_vector. Note the model (generator) training is set 'False'

    predictions = model(test_input,training=False)

    

    fig = plt.figure(figsize=(4,4))

    print('Epoch: ' + str(epoch))

    for i in range(predictions.shape[0]):

        plt.subplot(4,4,i+1)

        plt.imshow(predictions[i,:,:,0]*127.5 + 127.5, cmap='gray')

        plt.axis('off')

    

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    plt.show()
gantraining = tf.contrib.eager.defun(gantraining)
def theGAN(dataset, epochs):

    for epoch in range(epochs):

        for images in dataset:

            gantraining(images)

        

        display.clear_output(wait=True)

        image_processing(generator, epoch+1, random_vector)

        

        # saving (checkpoint) the model every 15 epochs

        if (epoch + 1) % 15 == 0:

            checkpt.save(file_prefix = chkpt_prefix)

    

    display.clear_output(wait=True)

    image_processing(generator, epoch, random_vector)
theGAN(traindata, EPOCHS)
checkpt.restore(tf.train.latest_checkpoint(chkpt_dir))
def display_image(epoch_no):

  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(EPOCHS)