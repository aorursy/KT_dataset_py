# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob
train_images_alder = glob.glob('/kaggle/input/trunk-assignment/alder/*.JPG')
train_images_ginkgo_biloba = glob.glob('/kaggle/input/trunk-assignment/ginkgo biloba/*.JPG')
train_images_birch = glob.glob('/kaggle/input/trunk-assignment/birch/*.JPG')
train_images_beech = glob.glob('/kaggle/input/trunk-assignment/beech/*.JPG')
train_images_chestnut = glob.glob('/kaggle/input/trunk-assignment/chestnut/*.JPG')
train_images_hornbeam = glob.glob('/kaggle/input/trunk-assignment/hornbeam/*.JPG')
train_images_horse_chestnut = glob.glob('/kaggle/input/trunk-assignment/horse chestnut/*.JPG')
train_images_linden = glob.glob('/kaggle/input/trunk-assignment/linden/*.JPG')
train_images_oak = glob.glob('/kaggle/input/trunk-assignment/oak/*.JPG')
train_images_pine = glob.glob('/kaggle/input/trunk-assignment/pine/*.JPG')
train_images_spruce = glob.glob('/kaggle/input/trunk-assignment/spruce/*.JPG')
train_images_oriental_plane= glob.glob('/kaggle/input/trunk-assignment/oriental plane/*.JPG')
train_images=[train_images_alder,train_images_ginkgo_biloba,train_images_birch,train_images_beech,train_images_chestnut,train_images_hornbeam ,train_images_horse_chestnut,train_images_linden,train_images_oak,train_images_pine,train_images_spruce,train_images_oriental_plane]
import cv2

y_train=[]
X_train = []
for (label, fnames) in enumerate(train_images):
    for fname in fnames:
            img = cv2.imread(fname)
            img = cv2.resize(img, (150 ,150 ) , interpolation=cv2.INTER_AREA)
            img=  img.astype('float32') / 255.
            y_train.append(label)
            X_train.append(img)
import numpy as np
X_train=np.array(X_train)
            
categories=['alder','ginkgo_biloba','birch','beech','chestnut','hornbeam','horse_chestnut','linden','oak','pine','spruce','oriental_plane']
import matplotlib.pyplot as plt
i = 0
plt.figure(figsize=(10,10))
plt.subplot(221,title=categories[y_train[i]]), plt.imshow(X_train[i], cmap='gray')
plt.subplot(222,title=categories[y_train[i+34]]), plt.imshow(X_train[i+34], cmap='gray')
plt.subplot(223,title=categories[y_train[i+200]]), plt.imshow(X_train[i+200], cmap='gray')
plt.subplot(224,title=categories[y_train[i+300]]), plt.imshow(X_train[i+300], cmap='gray')
from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, y_test = train_test_split(X_train, y_train, test_size=0.1,train_size=0.9,shuffle=True, stratify=y_train)
x_new_train, x_val, y_new_train, y_val = train_test_split(x_train, Y_train, test_size=0.2,train_size=0.8,shuffle=True, stratify=Y_train)

from keras.utils.np_utils import to_categorical

y_new_train= to_categorical(y_new_train)
y_val=to_categorical(y_val)
y_test=to_categorical(y_test)
print(x_new_train.shape)
print(x_val.shape)
print(x_test.shape)
print(y_new_train.shape)
print(y_val.shape)
print(y_test.shape)
batch_size = 128
no_epochs = 50
validation_split = 0.2
verbosity = 1
latent_dim = 2
num_channels = 3
input_shape=(150,150,3)
img_width, img_height = x.shape[1], x.shape[2]
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
i       = Input(shape=input_shape, name='encoder_input')
'''
cx      = Conv2D(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(i)
cx      = BatchNormalization()(cx)
cx      = Conv2D(filters=256, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv2D(filters=512, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv2D(filters=1024, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)'''

new = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
new = MaxPooling2D((2, 2), padding='same')(new)
new   = BatchNormalization()(new)
new = Conv2D(64, (3, 3), activation='relu', padding='same')(new)
new = MaxPooling2D((2, 2), padding='same')(new)
new   = BatchNormalization()(new)
new = Conv2D(128, (3, 3), activation='relu', padding='same')(new)
new = MaxPooling2D((2, 2), padding='same')(new)
cx   = BatchNormalization()(new)
x       = Flatten()(cx)
x       = Dense(20, activation='relu')(x)
x       = BatchNormalization()(x)
mu      = Dense(latent_dim, name='latent_mu')(x)
sigma   = Dense(latent_dim, name='latent_sigma')(x)
conv_shape = K.int_shape(cx)
# Define sampling with reparameterization trick
def sample_z(args):
    mu, sigma = args
    batch     = K.shape(mu)[0]
    dim       = K.int_shape(mu)[1]
    eps       = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps
# Use reparameterization trick to ensure correct gradient
z       = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])
encoder = Model(i, [mu, sigma, z], name='encoder')
encoder.summary()
d_i   = Input(shape=(latent_dim, ), name='decoder_input')
x     = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
x     = BatchNormalization()(x)
x     = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
'''
cx    = Conv2DTranspose(filters=1024, kernel_size=5, strides=2,padding='same', activation='relu')(x)
cx    = BatchNormalization()(cx)
cx    = Conv2DTranspose(filters=512, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
cx    = BatchNormalization()(cx)
cx    = Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
cx    = BatchNormalization()(cx)
cx    = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
cx    = BatchNormalization()(cx)
o     = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
'''
new = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
new = UpSampling2D((2, 2))(x)
new   = BatchNormalization()(new)
new = Conv2D(64, (3, 3), activation='relu', padding='same')(new)
new = UpSampling2D((2, 2))(new)
new   = BatchNormalization()(new)
new = Conv2D(32, (3, 3), activation='relu',padding='same')(new)
new = UpSampling2D((2, 2))(new)
new   = BatchNormalization()(new)
decoded = Conv2D(3, (3, 3), activation='sigmoid')(new)
decoder = Model(d_i, decoded, name='decoder')
decoder.summary()
vae_outputs = decoder(encoder(i)[2])
vae         = Model(i, vae_outputs, name='vae')
vae.summary()
def kl_reconstruction_loss(true, pred):
  # Reconstruction loss
   reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
  # KL divergence loss
   kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
   kl_loss = K.sum(kl_loss, axis=-1)
   kl_loss *= -0.5
  # Total loss = 50% rec + 50% KL divergence loss
   return K.mean(reconstruction_loss + kl_loss)
vae.compile(optimizer='Adam', loss=kl_reconstruction_loss)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=10)
history=vae.fit(x_new_train,x_new_train, batch_size = batch_size,epochs = no_epochs,shuffle=True,validation_data=(x_val,x_val), callbacks=[early_stopping_monitor])
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(12)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

def viz_latent_space(encoder, data):
    input_data, target_data = data
    mu, _, _ = encoder.predict(input_data)
    plt.figure(figsize=(8, 10))
    plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    plt.colorbar()
    plt.show()

def viz_decoded(encoder, decoder, data):
    num_samples = 15
    figure = np.zeros((img_width * num_samples, img_height * num_samples, num_channels))
    grid_x = np.linspace(-4, 4, num_samples)
    grid_y = np.linspace(-4, 4, num_samples)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(img_width, img_height, num_channels)
            figure[i * img_width: (i + 1) * img_width,j * img_height: (j + 1) * img_height] = digit
    plt.figure(figsize=(10, 10))
    start_range = img_width // 2
    end_range = num_samples * img_width + start_range + 1
    pixel_range = np.arange(start_range, end_range, img_width)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
  # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
  # So reshape if necessary
    fig_shape = np.shape(figure)
    if fig_shape[2] == 1:
        figure = figure.reshape((fig_shape[0], fig_shape[1]))
  # Show image
    plt.imshow(figure)
    plt.show()
data = (x_test, y_test)
viz_latent_space(encoder, data)
viz_decoded(encoder, decoder, data)