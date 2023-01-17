import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# imports used in this project

# keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
# keras layers
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
# ploting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import MNIST dataset and format the images
(x_train_imgs, _), (x_test_imgs, _) = mnist.load_data()
image_size = x_train_imgs.shape[1]
x_train = np.reshape(x_train_imgs, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test_imgs, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# format image back to ordinary image format
def imageShowFormat(img):
    image = img.reshape(image_size,image_size)
    return (image*255).astype(np.uint8)
# Create noised images
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise
# limit the noisy image's pixel to 0-1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# Plot some of the normal&noised images
fig = plt.figure(figsize=(8,8))
# select 8 images
img_indexes = np.random.choice(len(x_train),8)
for i,img_index in enumerate(img_indexes):
    original = x_train[img_index]
    noised = x_train_noisy[img_index]
    sp = plt.subplot(4,4,i*2+1)
    sp.set_title('original')
    sp.axis('off')
    plt.imshow(imageShowFormat(original))
    sp = plt.subplot(4,4,i*2+2)
    sp.set_title('noised')
    sp.axis('off')
    plt.imshow(imageShowFormat(noised))
# Encoder
# input layer
encoder_inputs = Input(shape=(image_size, image_size, 1), name='encoder_input')
# convolutioning and shrink down
encoder_1 = Conv2D(filters=32,kernel_size=3,strides=2,activation='relu',padding='same')(encoder_inputs)
encoder_2 = Conv2D(filters=64,kernel_size=3,strides=2,activation='relu',padding='same')(encoder_1)
# flatten and make latent
encoder_flat = Flatten()(encoder_2)
latent_length = 16
latent = Dense(latent_length, name='latent_vector')(encoder_flat)
# the encoder model
encoder = Model(encoder_inputs, latent, name='encoder')
encoder.summary()
# Decoder
# latent input
decoder_inputs = Input(shape=(latent_length,), name='decoder_input')
# a dense a layer to get enough output numbers
decoder_dense = Dense(encoder_2.shape[1]*encoder_2.shape[2]*encoder_2.shape[3])(decoder_inputs)
# reshape to images
decoder_reshape = Reshape((encoder_2.shape[1],encoder_2.shape[2],encoder_2.shape[3]))(decoder_dense)
# transpose the conv2d
decoder_bigger_1 = Conv2DTranspose(filters=64,kernel_size=3,strides=2,activation='relu',padding='same')(decoder_reshape)
decoder_bigger_2 = Conv2DTranspose(filters=32,kernel_size=3,strides=2,activation='relu',padding='same')(decoder_bigger_1)
decoder_single = Conv2DTranspose(filters=1,kernel_size=3,padding='same')(decoder_bigger_2)
# the decoded image
decoder_outputs = Activation('sigmoid', name='decoder_output')(decoder_single)
# the decoder model
decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
decoder.summary()
# AutoEncoder
autoencoder = Model(encoder_inputs, decoder(encoder(encoder_inputs)), name='autoencoder')
autoencoder.summary()
# callback for each epoch
model_path = '/kaggle/working/best_model.h5'
callbacks = [
    # save model
    ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
]
# compiling
autoencoder.compile(loss='mse', optimizer='adam')
# fitting
autoencoder.fit(x_train_noisy,x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=30,
                batch_size=128,
                callbacks=callbacks)
# load the best model
autoencoder.load_weights(model_path)
# predict
test_denoised = autoencoder.predict(x_test_noisy)
fig = plt.figure(figsize=(6,16))
# select 8 images
img_indexes = np.random.choice(len(x_test),8)
for i,img_index in enumerate(img_indexes):
    original = x_test[img_index]
    noised = x_test_noisy[img_index]
    denoised = test_denoised[img_index]
    sp = plt.subplot(8,3,i*3+1)
    sp.set_title('original')
    sp.axis('off')
    plt.imshow(imageShowFormat(original))
    sp = plt.subplot(8,3,i*3+2)
    sp.set_title('noised')
    sp.axis('off')
    plt.imshow(imageShowFormat(noised))
    sp = plt.subplot(8,3,i*3+3)
    sp.set_title('denoised')
    sp.axis('off')
    plt.imshow(imageShowFormat(denoised))