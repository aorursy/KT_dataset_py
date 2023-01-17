# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings("ignore") 
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.datasets import mnist
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.preprocessing.image import array_to_img
svnh_data = loadmat('/kaggle/input/train_32x32.mat')
svnhTrain_X = svnh_data['X']
svnhTrain_Y = svnh_data['y']
svnhTrain_X = np.rollaxis(svnhTrain_X, 3)
svnhTrain_X = rgb2gray(svnhTrain_X)
svnhTrain_X = np.expand_dims(svnhTrain_X, axis=3)
svnhTrain_X = svnhTrain_X[:60000]
svnhTrain_Y = svnhTrain_Y[:60000]
#svnhTrain_Y = np.concatenate(svnhTrain_Y)
#svnhTrain_Y = to_categorical(svnhTrain_Y)
(mnistx_train, mnisty_train), (mnistx_test, mnisty_test) = mnist.load_data()
def _mnistRe(imgs):
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs
mnistx_train = _mnistRe(mnistx_train)
def _featureModel():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='SAME', data_format='channels_last', 
                            input_shape=(32, 32, 1)))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(16, (3, 3), strides=(2, 2), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    #model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Activation('relu'))
#     #model.add(layers.Dropout(0.5))
#     model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
#     model.add(layers.BatchNormalization(momentum=0.9))
#     model.add(layers.Activation('relu'))
#     #model.add(layers.Dropout(0.5))
#     model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
#     model.add(layers.BatchNormalization(momentum=0.9))
#     model.add(layers.Activation('relu'))
#     model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
#     model.add(layers.BatchNormalization(momentum=0.9))
#     model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(16, (1, 1), strides=(2, 2), padding='VALID'))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation('tanh'))

#     model.add(layers.LeakyReLU(alpha=0.2)
    model.summary()
    
    return model
f = _featureModel()
def _dModel():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='SAME', data_format='channels_last', 
                            input_shape=(32, 32, 1)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(128, (2, 2), strides=(2, 2), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(1, (2, 2), strides=(2, 2), padding='VALID'))
    model.add(layers.BatchNormalization(momentum=0.95))
    #model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    #model.add(layers.Dense(784))
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU(alpha=0.01))
    #model.add(layers.Dense(1))
    #model.add(Activation('sigmoid'))
    #model.add(layers.Dense(1, activation='sigmoid'))
    
    model.summary()
    #_,discriminator = _dModel()
    #optimizer = Adam(0.0002, 0.5)
    #model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])
    
    return model 
d = _dModel()
def _gModel():
    model = tf.keras.Sequential()
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='VALID', data_format='channels_last', 
                            input_shape=(1, 1, 16)))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
#     model.add(layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same'))
#     #model.add(layers.BatchNormalization(momentum=0.95))
#     #model.add(layers.Activation('relu'))
#     #model.add(layers.Dropout(0.5))
#     model.add(layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same'))
#     model.add(layers.BatchNormalization(momentum=0.95))
#     #model.add(layers.Activation('relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))
#     model.add(layers.BatchNormalization(momentum=0.95))
#     #model.add(layers.Activation('relu'))
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='SAME'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('tanh'))
    
    model.add(layers.Conv2D(1, (3, 3), strides=(1, 1), padding='SAME'))
#     #model.add(Reshape((32, 32, 1)))
    model.summary()
    
    return model
g = _gModel()
def get_batch(data, batch_size):
    indices = np.random.randint(low=0, high=len(data), size=batch_size)
    np.random.shuffle(indices)
    batch_data = data[indices]
    return batch_data
def _results(losses):
    labels = ['Discriminator', 'Generator']
    losses = np.array(losses)    
    
    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.show()
    plt.savefig('Loss_Plot.png')
def _images(generated_images):
    n_images = len(generated_images)
    rows = 4
    cols = n_images//rows
    
    plt.figure(figsize=(cols, rows))
    for i in range(n_images):
        img = array_to_img(generated_images[i], data_format='channels_last', scale=True)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
    plt.savefig('Generated_Images.png')
def discriminator_loss(real_output, fakeMNIST_output, fakeSVHN_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fakeMNIST_loss = cross_entropy(tf.zeros_like(fakeMNIST_output), fakeMNIST_output)
    fakeMNIST_loss = cross_entropy(tf.zeros_like(fakeSVHN_output), fakeSVHN_output)
    
    total_loss = real_loss + fakeMNIST_loss + fakeMNIST_loss
    return total_loss
def generator_loss(fakeMNIST_output, fakeSVHN_output):
    g1_loss = cross_entropy(tf.ones_like(fakeMNIST_output), fakeMNIST_output)
    g2_loss = cross_entropy(tf.ones_like(fakeSVHN_output), fakeSVHN_output)
    
    return g1_loss + g2_loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#@tf.function
def train(epochs=100,
    batch_size=128,
    eval_size=16,    
    _details=True):
    
    losses = []
    encoder, generator, discriminator = f, g, d
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-6)
    
    for e in range(epochs):
        for i in range(int(svnhTrain_X.shape[0]/batch_size)):
            mnist_samples = get_batch(mnistx_train, batch_size=batch_size)
            svhn_samples = get_batch(svnhTrain_X, batch_size=batch_size)
            
            mnist_samples = tf.convert_to_tensor(mnist_samples, float)
            svhn_samples = tf.convert_to_tensor(svhn_samples, float)
            
            with tf.GradientTape() as gTape, tf.GradientTape() as dTape:
                encoded_SVHN = encoder(svhn_samples)
                encoded_MNIST = encoder(mnist_samples)
                
                generatedSVHN_images = generator(encoded_SVHN)
                generatedMNIST_images = generator(encoded_MNIST)
                SVHN_fgx = encoder(generatedSVHN_images)
                    
                real_out = discriminator(mnist_samples)
                fake_SVHNdiscOut = discriminator(generatedSVHN_images)
                fake_MNISTdiscOut = discriminator(generatedMNIST_images)
                
                d_loss = (discriminator_loss(real_out, fake_MNISTdiscOut, fake_SVHNdiscOut))
                
                g1_loss = generator_loss(generatedMNIST_images, generatedSVHN_images)
                g2_loss = tf.reduce_mean(tf.square(encoded_SVHN - SVHN_fgx))*15
                gSmoothing_1 = tf.reduce_mean(tf.square(mnist_samples - generatedMNIST_images))
                gSmoothing_2 = tf.reduce_mean(tf.math.squared_difference(generatedMNIST_images[:,1:,:,:], 
                                                                    generatedMNIST_images[:,:-1,:,:]))+tf.reduce_mean(
                    tf.math.squared_difference(generatedMNIST_images[:,:,1:,:], generatedMNIST_images[:,:,:-1,:]))+tf.reduce_mean(
                    tf.math.squared_difference(generatedSVHN_images[:,1:,:,:], generatedSVHN_images[:,:-1,:,:]))+tf.reduce_mean(
                    tf.math.squared_difference(generatedSVHN_images[:,:,1:,:], generatedSVHN_images[:,:,:-1,:]))
                
                g_loss = g1_loss + (15*g2_loss) + (15*gSmoothing_1) + gSmoothing_2
                
            gGradients = gTape.gradient(g_loss, generator.trainable_variables)
            dGradients = dTape.gradient(d_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gGradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(dGradients, discriminator.trainable_variables))
            losses.append((d_loss, g_loss))
        
        g_eval = get_batch(svnhTrain_X, batch_size=eval_size)
        g_encEval = encoder.predict_on_batch(g_eval)
        eval_fake = generator.predict_on_batch(g_encEval)

        print("Epoch:{:>3}/{} Discriminator Loss:{:>7.4f} Generator Loss:{:>7.4f}".format(
            e+1, epochs, d_loss, g_loss))
        
        if _details and (e+1)%5==0:
            print("SVNH Images to transfer", '\n', _images(g_eval))
            print("SVHN-MNIST Transfered Images", '\n', _images(eval_fake))
            
    if _details:
        _results(losses)
    return generator
train()