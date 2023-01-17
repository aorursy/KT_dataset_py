import tensorflow as tf

#tf.compat.v1.disable_eager_execution()

from tensorflow.keras.utils import plot_model

from IPython.display import SVG

from tensorflow.keras.utils import model_to_dot

import os

import numpy as np

from numpy import array

import cv2

import urllib.request

import keras.backend as K

import matplotlib.pyplot as plt

from PIL import Image

from keras import regularizers

from keras.layers import Dense

from keras.layers import Input

from keras.layers import MaxPool2D

from keras.layers import Input

from keras.layers import add

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.convolutional import UpSampling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.models import Model

from keras.optimizers import Adam

from keras.applications.vgg19 import VGG19

from keras.applications import MobileNet

from keras.models import load_model



tf.config.experimental_run_functions_eagerly(True)



config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session()
class VGG_LOSS(object):



    def __init__(self, imshape):

        

        self.image_shape = imshape



    # computes VGG loss or content loss

    def vgg_loss(self, y_true, y_pred):

    

        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)

        vgg19.trainable = False

        # Make trainable as False

        for l in vgg19.layers:

            l.trainable = False

        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)

        model.trainable = False

    

        return K.mean(K.square(model(y_true) - model(y_pred)))
# Normalize/Denormalize images

def normalize(input_data):

    return input_data.astype(np.float32)/255



def denormalize(input_data):

    input_data = input_data *255

    return input_data.astype(np.uint8)
def get_optimizer(): 

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    return adam
def load_data(path):

    real_image_treat_as_y = []

    downsize_image_treat_as_x = []

    for dirname, _, filenames in os.walk(path):

        for filename in filenames:       

            image = cv2.imread(os.path.join(dirname, filename))

            reshaped_image = cv2.resize(image, (256, 256))

            

            if reshaped_image.shape[-1] == 3:

                real_image_treat_as_y.append(reshaped_image)

            image = cv2.resize(image, (64, 64))

            reshaped_image = cv2.resize(image, (256, 256))

            

            if reshaped_image.shape[-1] == 3:

                downsize_image_treat_as_x.append(cv2.resize(image, (256, 256)))

    return (np.array(downsize_image_treat_as_x), np.array(real_image_treat_as_y))
downized_images, real_images = load_data('../input/ohgodno')
y_train = normalize(real_images[:800])

x_train = normalize(downized_images[:800])



y_test = normalize(real_images[800:900])

x_test = normalize(downized_images[800:900])
print(x_train.max())

print(x_train.min())

print(x_test.max())

print(x_test.min())

print(y_train.max())

print(y_train.min())

print(y_test.max())

print(y_test.min())
plt.figure(figsize=(15, 15))

plt.subplot(1, 2, 1)

plt.imshow(x_train[1])

plt.subplot(1, 2, 2)

plt.imshow(y_train[1])
imshape = (256, 256, 3)

loss = VGG_LOSS(imshape)

optimizer = get_optimizer()
input_img = Input(shape=imshape)



l1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)

l2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)

l3 = MaxPool2D(padding='same')(l2)



l4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l3)

l5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)

l6 = MaxPool2D(padding='same')(l5)



l7 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l6)



l8 = UpSampling2D()(l7)

l9 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l8)

l10 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l9)



l11 = add([l10, l5])



l12 = UpSampling2D()(l11)

l13 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l12)

l14 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l13)



l15 = add([l14, l2])



decoded_image = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l15)



auto_encoder = Model(inputs=(input_img), outputs=decoded_image)
# auto_encoder.compile(optimizer=optimizer, loss=[loss.vgg_loss,'binary_crossentropy'], loss_weights=[1., 1e-3])#

auto_encoder.compile(optimizer=optimizer, loss='mean_squared_error')
auto_encoder.summary()
hist = auto_encoder.fit(x_train, y_train, epochs=1, batch_size=32, shuffle=True, validation_split=0.15)
a = auto_encoder.predict(x_test[:10])
a[1].max()
plt.figure(figsize=(15, 15))

plt.subplot(1, 3, 1)

plt.imshow(x_test[1])

plt.subplot(1, 3, 2)

plt.imshow(a[1])

plt.subplot(1, 3, 3)

plt.imshow(y_test[1])
b = a[1] - y_test[1]

plt.imshow(b)
s = denormalize(a[1])
auto_encoder.save('autoencoder.h5')
#auto_encoder2 = load_model('../input/newmodelnewhope/autoencoder_vgg.h5', custom_objects={'vgg_loss': loss.vgg_loss})
auto_encoder2.fit(x_train, y_train, epochs=100, batch_size=8, shuffle=True, validation_split=0.15)
a = auto_encoder2.predict(x_test[:10])
plt.figure(figsize=(15, 15))

plt.subplot(1, 3, 1)

plt.imshow(x_test[1])

plt.subplot(1, 3, 2)

plt.imshow(a[1])

plt.subplot(1, 3, 3)

plt.imshow(y_test[1])
b = a[1] - y_test[1]

plt.imshow(b)