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
import cv2

import glob

import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train = []

files = glob.glob("../input/train/*.jpg") # your image path

for myFile in files:

    train_img = load_img(myFile,target_size=(150,150),color_mode="rgb")

    train_img= img_to_array(train_img)

    train.append(train_img)



train_tar = []

for myFile in files:

    train_img = load_img(myFile,target_size=(150,150),color_mode="grayscale")

    train_img= img_to_array(train_img)

    train_tar.append(train_img)

    

test = []

fikes = glob.glob("../input/test/*.jpg")

for myfile in fikes:

    test_img = load_img(myFile,target_size=(150,150),color_mode="rgb")

    test_img= img_to_array(test_img)

    test.append(test_img)



print("Length of Train is:",len(train))

print("Length of Test is:",len(test))
train = np.asarray(train)

print("The Shape of Train array is",train.shape)



train_tar = np.asarray(train_tar)

print("The Shape of Train grayscale array is",train_tar.shape)



test = np.asarray(test)

print("The Shape of Test array is",test.shape)
import keras

from keras.models import Model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Reshape

from keras import optimizers

from tensorflow.keras import backend as K



# creating autoencoder model

input_img = Input(shape = (150,150,3))

latent_dim = 256



#encoder



conv1 = Conv2D(64, (2, 2), activation='relu', padding='same')(input_img) 

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

conv2 = Conv2D(256, (2, 2), activation='relu', padding='same')(pool1) 

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 

conv3 = Conv2D(512, (2, 2), activation='relu', padding='same')(pool2)

encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)



shape = K.int_shape(encoded)



# generate a latent vector

x = Flatten()(encoded)

latent = Dense(latent_dim, name='latent_vector')(x)



# instantiate encoder model

encoder = Model(input_img, latent, name='encoder')

encoder.summary()





#decoder

latent_inputs = Input(shape=(latent_dim,), name='decoder_input')

x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)

x = Reshape((shape[1], shape[2], shape[3]))(x)



conv4 = Conv2D(512,(2, 2), activation='relu', padding='same')(x) 

up1 = UpSampling2D((2,2))(conv4) 

conv5 = Conv2D(256,(2, 2), activation='relu', padding='same')(up1) 

up2 = UpSampling2D((2,2))(conv5) 

conv6 = Conv2D(64,(2, 2), activation='relu', padding='same')(up2) 

up3 = UpSampling2D((2,2))(conv6) 

decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(up3)



# instantiate decoder model

decoder = Model(latent_inputs, decoded, name='decoder')

decoder.summary()







#Fitting the model

autoencoder = Model(input_img, decoder(encoder(input_img)))

adam = optimizers.Adam(learning_rate=0.001,amsgrad=False)

autoencoder.compile(optimizer=adam, loss='mean_squared_error')

autoencoder.summary()
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),

                               cooldown=0,

                               patience=5,

                               verbose=1,

                               min_lr=0.5e-6)

# called every epoch

callbacks = [lr_reducer]



autoencoder.fit(train,train_tar,

                epochs=5,

                validation_split=0.20,

                verbose =1

                #callbacks=callbacks

                )
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt

decoded_imgs = autoencoder.predict(test)



n = 10

plt.figure(figsize=(20, 4))

for i in range(1,n+1):

    # display original

    ax = plt.subplot(2, n, i)

    img = array_to_img(test[i])

    plt.imshow(img)

    #plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + n)

    img_dec = array_to_img(decoded_imgs[i])

    plt.imshow(img_dec)

    #plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
plt.imshow(array_to_img(train[8]))
plt.imshow(array_to_img(train_tar[8]))
import cv2

#from skimage import transform



train_new = []

files = glob.glob("../input/train/*.jpg") # your image path

for myFile in files:

    train_img = cv2.imread(myFile,0)

    #train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    train_img = cv2.resize(train_img,(128,128))

    train_img = np.expand_dims(train_img, axis=2)

    train_new.append(train_img)

n = 10

plt.figure(figsize=(20, 4))

for i in range(1,n+1):

    # display original

    ax = plt.subplot(2, n, i)

    img = array_to_img(train_new[i])

    plt.imshow(img)

    #plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)