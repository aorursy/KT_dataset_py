# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ds_dir = '/kaggle/input/kannada-mnist/kannada_mnist_datataset_paper/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST'



X_train = np.load(os.path.join(ds_dir,'X_kannada_MNIST_train.npz'))['arr_0']

X_test = np.load(os.path.join(ds_dir,'X_kannada_MNIST_test.npz'))['arr_0']

y_train = np.load(os.path.join(ds_dir,'y_kannada_MNIST_train.npz'))['arr_0']

y_test = np.load(os.path.join(ds_dir,'y_kannada_MNIST_test.npz'))['arr_0']



print(X_train.shape, X_test.shape)

print(y_train.shape, y_test.shape)
# Reshaping for MLPs

X_train = X_train.astype('float32') / 255.

X_test = X_test.astype('float32') / 255.



X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))

X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

print( X_train.shape)

print( X_test.shape)
def plot_n(X,y, n=10, title = ""):

    plt.figure(figsize=(20, 4)) 

    for i in range(n):

        ax = plt.subplot(2, n, i + 1)

        plt.imshow(X[i].reshape(28, 28))

        plt.title(y[i])

        plt.gray()

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    plt.suptitle(title)

    



plot_n(X_train, y_train,  5, "Original Data")
from keras.layers import Input, Dense, UpSampling2D, Conv2D, MaxPooling2D

from keras.models import Model

from keras import regularizers
encoding_dim = 32

input_img = Input(shape = (784,))

encoded = Dense(encoding_dim, activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(input_img)

decoded = Dense(784, activation= 'relu')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape = (encoding_dim,) )

decoded_layer = Dense(784, activation = 'relu')(encoded_input)

decoder = Model(encoded_input, decoded_layer)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, 

               epochs = 50, 

               batch_size=256, 

               shuffle=True, 

               validation_data=(X_test, X_test))
regen_test_1 = autoencoder.predict(X_test)
plot_n(regen_test_1, y_test, 5)

plot_n(X_test, y_test, 5)
encoding_dim = 32



input_img = Input(shape = (784,))

encoded = Dense(128, activation = 'relu' )(input_img)

encoded = Dense(64, activation = 'relu')(encoded)

encoded = Dense(32, activation = 'relu')(encoded)



decoded = Dense(64, activation = 'relu')(encoded)

decoded = Dense(128, activation = 'relu')(decoded)

decoded = Dense(784, activation = 'relu')(decoded)





autoencoder_2 = Model(input_img, decoded)
autoencoder_2.compile(optimizer='adadelta', loss='binary_crossentropy')

history_2 = autoencoder_2.fit(X_train, X_train, 

               epochs = 200, 

               batch_size=256, 

               shuffle=True, 

               validation_data=(X_test, X_test))
regen_test_2 = autoencoder_2.predict(X_test)
plot_n(regen_test_1, y_test, 5, "Single MLP")

plot_n(regen_test_2, y_test, 5, "Deep MLP")

plot_n(X_test, y_test, 5, "Original Data")
encoding_dim = 32



input_img = Input(shape = (28,28,1))



x =  Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x =  Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x =  Conv2D(32, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2,2), padding= 'same')(x)



x =  Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2,2))(x)

x =  Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x =  Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2,2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)





autoencoder_3 = Model(input_img, decoded)
autoencoder_3.summary()
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1)) 

X_test = np.reshape(X_test, (len(X_test), 28, 28, 1)) 
autoencoder_3.compile(optimizer='adadelta', loss='binary_crossentropy')

history_3 = autoencoder_3.fit(X_train, X_train, 

               epochs = 100, 

               batch_size=256, 

               shuffle=True,

               validation_data=(X_test, X_test))
regen_test_3 = autoencoder_3.predict(X_test)
plot_n(regen_test_1, y_test, 5, "Single layer MLP")

plot_n(regen_test_2, y_test, 5, "Deep MLP")

plot_n(regen_test_3, y_test, 5, "CNN")

plot_n(X_test, y_test, 5, "Original")
# Adding some normal noise to the data. 

noise_factor = 0.3

X_train_noise = X_train + noise_factor* np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

X_test_noise = X_test + noise_factor* np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 



X_train_noise = np.clip(X_train_noise, 0., 1.)

X_test_noise = np.clip(X_test_noise, 0., 1.)
plot_n(X_train, y_train, 5, "Original Data")

plot_n(X_train_noise, y_train, 5, "Noisy Added")
encoding_dim = 32



input_img = Input(shape = (28,28,1))



x =  Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x =  Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x =  Conv2D(32, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2,2), padding= 'same')(x)



x =  Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2,2))(x)

x =  Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x =  Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2,2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)





autoencoder_4 = Model(input_img, decoded)
# autoencoder_3.summary()
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1)) 

X_test = np.reshape(X_test, (len(X_test), 28, 28, 1)) 
autoencoder_3.compile(optimizer='adadelta', loss='binary_crossentropy')

history_3 = autoencoder_3.fit(X_train_noise, X_train, 

               epochs = 100, 

               batch_size=256, 

               shuffle=True,

               validation_data=(X_test_noise, X_test))
regen_test_3 = autoencoder_3.predict(X_test_noise)
plot_n(regen_test_3, y_test, 5, "Regenrated denoised")

plot_n(X_test_noise, y_test, 5, "Input Noisy")

plot_n(X_test, y_test, 5, "Input")