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
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train.shape
import numpy as np
from skimage import transform

def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs
X_train_mn = resize_batch(X_train)
from keras.preprocessing.image import ImageDataGenerator, array_to_img
print(X_train_mn.shape)
array_to_img(X_train_mn[1])
#print(trainX.shape)[]
from scipy.io import loadmat
from skimage.color import rgb2gray
def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']

X_train_sv, y_train_sv = load_data('/kaggle/input/train_32x32.mat')
X_test_sv, y_test_sv = load_data('/kaggle/input/test_32x32.mat')
X_train_sv = np.rollaxis(X_train_sv, 3)
X_train_sv = rgb2gray(X_train_sv)
X_train_sv = np.expand_dims(X_train_sv, axis=3)

# Transpose the image arrays
#X_train_sv, y_train_sv = X_train_sv.transpose((3,0,1,2)), y_train_sv[:,0]
X_test_sv, y_test_sv = X_test_sv.transpose((3,0,1,2)), y_test_sv[:,0]

print("Training Set", X_train_sv.shape, y_train_sv.shape)
print("Test Set", X_test_sv.shape, y_test_sv.shape)
#mnist 3 channels
from keras import backend as K

def grayscale_to_rgb(images, channel_axis=-1):
    #images= K.expand_dims(images, axis=channel_axis)
    tiling = [1] * 4    # 4 dimensions: B, H, W, C
    tiling[channel_axis] *= 3
    images= K.tile(images, tiling)
    return images

target_X = grayscale_to_rgb(X_train_mn)
print(target_X.shape)
from keras.preprocessing.image import ImageDataGenerator, array_to_img
array_to_img(target_X[2])
# data normalization
def data_normalize(data):
    numerator = data - np.min(data, 0)
    #denominator = 255
    denominator = np.max(data, 0) - np.min(data, 0)
    
    return (numerator * (denominator))


def color_img_normalize(data):
    batch_size = len(data)
    row_size = len(data[0])
    col_size = len(data[0][0])
    channel_size = 3
    tmp = np.zeros((batch_size, row_size, col_size, channel_size), dtype=np.float32)
    for i in range(batch_size):
        tmp[i] = data[i]/255.0
    return tmp

#source_x = (X_train_sv - 127.5) / 127.5 # Normalize the images to [-1, 1]
#target_x = (target_X - 127.5) / 127.5 

source_x = (X_train_sv)
target_x = (X_train_mn)

source_x.shape
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import keras.backend as k
import tensorflow as tf
tf.__version__
def discriminator(in_shape=(32,32,1)):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64,(3,3),strides=(2,2),padding='same', input_shape=in_shape))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3,3),strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(8, (3,3),strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(1, (4,4),strides=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Flatten())
    net = model.add(layers.Activation('sigmoid'))
    #model.compile(optimizer=opt,loss='binary_crossentropy')
    return model

d_model = discriminator()
d_model.summary()
def feature_extractor():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64,(3,3),strides=(2,2),padding='same', input_shape=(32,32,1)))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128,(3,3),strides=(2,2),padding='same'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256,(3,3),strides=(2,2),padding='same'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128,(4,4),strides=(2,2),padding='valid'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('tanh'))
    model.summary()
    return  model
f_model = feature_extractor()
def generator(in_shape=(1,1,128)):
    model = tf.keras.Sequential()
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='valid',
                            input_shape=(1, 1, 128) ))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.95))
    model.add(layers.Activation('tanh'))
    model.summary()
    
    return model
g_model = generator()
def deprocess(x):
    x = (x / 2 + 1) * 255
    x = np.clip(x, 0, 255)
    x = np.uint8(x)
    x = x.reshape(32, 32)
    return x
def _images(generated_images):
    n_images = len(generated_images)
    rows = 4
    cols = n_images//rows
    
    plt.figure(figsize=(cols, rows))
    for i in range(n_images):
        #img = deprocess(generated_images[i])
        img = array_to_img(generated_images[i])
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        #plt.imshow(img[i, :, :, 0] * 255.0 , cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output1, fake_output2):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss_svhn = cross_entropy(tf.zeros_like(fake_output1), fake_output1)
    fake_loss_mnist = cross_entropy(tf.zeros_like(fake_output2), fake_output2)
    total_loss = real_loss + fake_loss_svhn + fake_loss_mnist
    return total_loss
def get_batch(data, batch_size):
    indices = np.random.randint(low=0, high=len(data), size=batch_size)
    np.random.shuffle(indices)
    batch_data = data[indices]
    return batch_data
def generator_loss(fake_output1,fake_output2):
    loss1 = cross_entropy(tf.ones_like(fake_output1), fake_output1)
    loss2 = cross_entropy(tf.ones_like(fake_output2), fake_output2)
    total_loss = loss1+loss2
    return total_loss
@tf.function
def train():
    epochs = 10
    batch_size=32
    eval_size=16
    _details = True

    encoder, generator, discriminator = f_model, g_model, d_model
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    

    losses = []
    for epoch in range(epochs):
        for i in range(int(target_x.shape[0]/batch_size)):
            X_batch_real = get_batch(target_x, batch_size=batch_size)
           
            g_samples_svhn = get_batch(source_x, batch_size=batch_size)
            g_samples_mnist = get_batch(target_x, batch_size=batch_size)
            X_batches_real = tf.convert_to_tensor(X_batch_real, float)
            g_samples_svhn = tf.convert_to_tensor(g_samples_svhn, float)
            g_samples_mnist = tf.convert_to_tensor(g_samples_mnist,float)

            with tf.GradientTape() as gTape, tf.GradientTape() as dTape:
                encoded = encoder(g_samples_svhn)
                generated_svhn = generator(encoded, training=True)
                src_fgfx = encoder(generated_svhn)

                encoded2 = encoder(g_samples_mnist)
                generated_mnist = generator(encoded2, training=True)

                real_out = discriminator(X_batches_real, training=True)
                fake_out1 = discriminator(generated_svhn, training=True)
                fake_out2 = discriminator(generated_mnist, training=True)

                
                L_Gang = generator_loss(fake_out1,fake_out2)
                L_const =  tf.reduce_mean(tf.square(encoded - src_fgfx))*15
                L_tid = tf.reduce_mean(tf.square(g_samples_mnist - generated_mnist))
                L_tv = tf.reduce_mean(tf.math.squared_difference(generated_mnist[:,1:,:,:], generated_mnist[:,:-1,:,:])) \
                            + tf.reduce_mean(tf.math.squared_difference(generated_mnist[:,:,1:,:], generated_mnist[:,:,:-1,:]))\
                            + tf.reduce_mean(tf.math.squared_difference(generated_svhn[:,1:,:,:], generated_svhn[:,:-1,:,:]))\
                            + tf.reduce_mean(tf.math.squared_difference(generated_svhn[:,:,1:,:], generated_svhn[:,:,:-1,:]))

                g_loss = L_Gang + (15* L_const) + (15*L_tid) + L_tv

                d_loss = discriminator_loss(real_out,fake_out1,fake_out2)

                gGradients = gTape.gradient(g_loss, generator.trainable_variables)
                dGradients = dTape.gradient(d_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gGradients,generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(dGradients,discriminator.trainable_variables))

        g_eval = get_batch(source_x,batch_size=eval_size)
        g_eval = encoder.predict_on_batch(g_eval)
        X_gen = generator.predict_on_batch(g_eval)

        print("Epoch:{:>3}/{} Discriminator Loss:{:>7.4f} Generator Loss:{:>7.4f}".format(
            epoch+1, epochs, d_loss, g_loss))
        
        
        
        if _details and (epoch+1)%1==0:
            _images((X_gen))
    return generator
gen = train()
x = target_x[0]
x = np.expand_dims(x, axis=0)
x.shape
plt.imshow(x[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
