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
#Importing all the dependencies

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

import keras

from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Reshape, Flatten, Lambda, Conv2DTranspose,UpSampling2D

from keras.preprocessing import backend as K

import matplotlib.pyplot as plt

import numpy as np

from keras.datasets import mnist

from keras import optimizers
# download training and test data from mnist and reshape it



(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.

output_X_train = X_train.reshape(-1,28,28,1)



X_test = X_test.astype('float32') / 255.

output_X_test = X_test.reshape(-1,28,28,1)



print(X_train.shape, X_test.shape)
# adding some noise to data



input_x_train = output_X_train + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=output_X_train.shape) 

input_x_test = output_X_test + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=output_X_test.shape)
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format



x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)



# at this point the representation is (7, 7, 32)



x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)

m = 128

n_epoch = 50

adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

autoencoder.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])

hist = autoencoder.fit(input_x_train,output_X_train, epochs=n_epoch, batch_size=m, shuffle=True,validation_split=0.20)
decoded_imgs = autoencoder.predict(input_x_test,verbose=0)



# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()
# summarize history for accuracy

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()
n = 15

plt.figure(figsize=(20, 4))

for i in range(1,n):

    # display original noisy image

    ax = plt.subplot(2, n, i)

    plt.imshow(input_x_test[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display denosined image

    ax = plt.subplot(2, n, i + n)

    plt.imshow(decoded_imgs[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()