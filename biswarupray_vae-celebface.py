import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import sklearn

from collections import Counter

import glob

import pickle

import tensorflow

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Input, Reshape

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D

from mpl_toolkits.axes_grid1 import AxesGrid

from sklearn.metrics import confusion_matrix

from tensorflow.keras import backend as K

from tensorflow.keras.losses import mse, binary_crossentropy

from tensorflow.keras.optimizers import SGD, Adam

import random

from random import shuffle

import skimage

import random

from skimage.io import imread



train = True

%matplotlib inline
pic_size = 64

def load_data_set(df, n):

    pics, labels = [], []

    i = 0

    for pic in df['image_id']:

        if i > n:

            break

        else:

            i+=1

        pic_url = "../input/celeba-dataset/img_align_celeba/img_align_celeba/"+pic

        temp = cv2.imread(pic_url)

        temp = cv2.resize(temp, (pic_size,pic_size)).astype('float32') / 255.

        pics.append(temp)

        labels.append(df[df['image_id'] == pic].values)

    X = np.array(pics)

    y = np.array(labels)

    y = y.reshape(y.shape[0], y.shape[2])

    print("Data set", X.shape, y.shape)

    return X, y
import pandas as pd



attr = pd.read_csv("../input/celeba-dataset/list_attr_celeba.csv")

attr.head()

feature_dict = {k:v for v,k in enumerate(attr.columns)}

n = 10000

X, y = load_data_set(attr, n)



print(feature_dict)
def show_sample_image(nb=3, df=attr, verbose=True):

    f, ax = plt.subplots(1, nb, figsize=(10,5))

    for i in range(nb):

        idx = random.randint(0, df.shape[0]-1)

        img_id = df.loc[idx].image_id

        img_uri = '../input/celeba-dataset/img_align_celeba/img_align_celeba/' + img_id

        img = skimage.io.imread(img_uri)  

        if verbose:

            label = img_id

            for col in df.columns:

                if df.loc[idx][col]==1:

                    label = label + '\n' + col  

            if nb > 1:

                ax[i].imshow(img)

                ax[i].set_title(label)

            else:

                ax.imshow(img) 

                ax.set_title(label)

        

    return img, list(df.loc[idx][1:df.shape[1]])
sample_img, sample_img_meta = show_sample_image()
#function to create batches of the large dataset

def sampling(args):

    """Reparameterization function by sampling from an isotropic unit Gaussian.

    # Arguments:

        args (tensor): mean and log of variance of Q(z|X)

    # Returns:

        z (tensor): sampled latent vector

    """



    z_mean, z_log_var = args

    batch = K.shape(z_mean)[0]

    dim = K.int_shape(z_mean)[1]

    # by default, random_normal has mean=0 and std=1.0

    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon



def create_VAE(input_shape):

    #CREATING THE VAE NETWORK

    image_size = input_shape[1]

    original_dim = image_size * image_size

    inputs = Input(shape=input_shape)

    print(inputs.shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    print(x.shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    x = AveragePooling2D(pool_size=(2, 2))(x)

    print(x.shape)

    x = Dropout(0.2)(x)

    

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    x = AveragePooling2D(pool_size=(2, 2))(x)

    print(x.shape)

    x = Dropout(0.2)(x)

    

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    x = AveragePooling2D(pool_size=(2, 2))(x)

    print(x.shape)

    x = Dropout(0.2)(x)

    

    x = Flatten()(x)

    print(x.shape)

    x = Dense(pic_size, activation='relu')(x)

    print(x.shape)

    

    latent_dim = 20

    

    latent_mean = Dense(latent_dim)(x)

    print(latent_mean.shape)

    latent_log_variance = Dense(latent_dim)(x)

    print(latent_log_variance.shape)

    

    latent_sample = Lambda(sampling)([latent_mean, latent_log_variance])

    print(latent_sample.shape)

    

    #BUILDING THE ENCODER

    encoder = Model(inputs, [latent_mean, latent_log_variance, latent_sample])

    

    latent_inputs = Input(shape=(latent_dim,))

    print(latent_inputs.shape)

    x = Dense(8*8*pic_size, activation='relu')(latent_inputs)

    print(x.shape)

    x = Reshape((8,8,pic_size))(x)

    print(x.shape)

    x = UpSampling2D((2, 2))(x)

    print(x.shape)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    

    x = UpSampling2D((2, 2))(x)

    print(x.shape)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    

    x = UpSampling2D((2, 2))(x)

    print(x.shape)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    print(x.shape)

   

    # Outputs for the encoder by a sigmoid activation function

    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    print(outputs.shape)

    

    #BUILDING THE DECODER

    decoder = Model(latent_inputs, outputs)

    

    # The outputs of the decoder are taken as the inputs of the encoder

    outputs = decoder(encoder(inputs)[2])

    vae = Model(inputs, outputs)

    

    # The losses of the VAE model is taken as the sum of the (binary cross entropy *dimensions) and the kl divergence loss

    reconstruction_loss = binary_crossentropy(inputs, outputs) * original_dim

    reconstruction_loss = K.mean(reconstruction_loss)

    print(reconstruction_loss)

    

    kl_loss = 1 + latent_log_variance - K.square(latent_mean) - K.exp(latent_log_variance)

    kl_loss = K.sum(kl_loss, axis=-1)

    kl_loss *= -0.5

    print(kl_loss)

    vae_loss = K.mean(reconstruction_loss + kl_loss)

    print(vae_loss)

    vae.add_loss(vae_loss)

    return vae, encoder, decoder
vae, encoder, decoder = create_VAE(input_shape=(pic_size,pic_size,3))

vae.compile(optimizer='adam', metrics=['accuracy'])

if train:

    epochs = 100

    for i in range(0,epochs):

        print(i)

        vae.fit(X, batch_size=32, epochs=1, verbose = 0)

        vae.save_weights("CELEBVAE.tf")

        encoder.save_weights("CELEBVAEE.tf")

        decoder.save_weights("CELEBVAED.tf")

    vae.fit(X, batch_size=32, epochs=1, verbose = 1)

from sklearn.decomposition import PCA



image_test = X[6]

image_reconstruction = vae.predict(np.expand_dims(image_test, axis = 0))[0]



z = np.array(encoder.predict(X)[2])



pca = PCA(n_components=20).fit(z)

z_pca = pca.transform(z)

z_female = z_pca[y[:,feature_dict['Male']] == -1]

z_avg_female_pca = np.mean(z_female, axis = 0)

X_avg_female = decoder.predict(np.expand_dims(pca.inverse_transform(z_avg_female_pca), axis = 0))[0]



z_male = z_pca[y[:,feature_dict['Male']] == 1]

z_avg_male_pca = np.mean(z_male, axis = 0)

X_avg_male = decoder.predict(np.expand_dims(pca.inverse_transform(z_avg_male_pca), axis = 0))[0]



z_glasses = z_pca[y[:,feature_dict['Eyeglasses']] == 1]

z_avg_glasses_pca = np.mean(z_glasses, axis = 0)

X_avg_glasses = decoder.predict(np.expand_dims(pca.inverse_transform(z_avg_glasses_pca), axis = 0))[0]



z_not_glasses = z_pca[y[:,feature_dict['Eyeglasses']] == -1]

z_avg_not_glasses_pca = np.mean(z_not_glasses, axis = 0)



z_reconstruction_w_glasses_pca = z_pca[6] + z_avg_glasses_pca - z_avg_not_glasses_pca

X_reconstruction_w_glasses = decoder.predict(np.expand_dims(pca.inverse_transform(z_reconstruction_w_glasses_pca), axis = 0))[0]



z_reconstruction_not_man_pca = z_pca[6] - z_avg_male_pca

X_reconstruction_not_man_pca = decoder.predict(np.expand_dims(pca.inverse_transform(z_reconstruction_not_man_pca), axis = 0))[0]



z_reconstruction_woman_pca = z_reconstruction_not_man_pca + z_avg_female_pca

X_reconstruction_woman_pca = decoder.predict(np.expand_dims(pca.inverse_transform(z_reconstruction_woman_pca), axis = 0))[0]



fig  = plt.figure(figsize=(20,20))



plt.subplot(2,4,1)

plt.imshow(cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB))

plt.subplot(2,4,2)

plt.imshow(cv2.cvtColor(image_reconstruction, cv2.COLOR_BGR2RGB))

plt.subplot(2,4,3)

plt.imshow(cv2.cvtColor(X_avg_female, cv2.COLOR_BGR2RGB))

plt.subplot(2,4,4)

plt.imshow(cv2.cvtColor(X_avg_male, cv2.COLOR_BGR2RGB))

plt.subplot(2,4,5)

plt.imshow(cv2.cvtColor(X_avg_glasses, cv2.COLOR_BGR2RGB))

plt.subplot(2,4,6)

plt.imshow(cv2.cvtColor(X_reconstruction_w_glasses, cv2.COLOR_BGR2RGB))

plt.subplot(2,4,7)

plt.imshow(cv2.cvtColor(X_reconstruction_not_man_pca, cv2.COLOR_BGR2RGB))

plt.subplot(2,4,8)

plt.imshow(cv2.cvtColor(X_reconstruction_woman_pca, cv2.COLOR_BGR2RGB))
vae.save_weights("CELEBVAE.tf")

encoder.save_weights("CELEBVAEE.tf")

decoder.save_weights("CELEBVAED.tf")