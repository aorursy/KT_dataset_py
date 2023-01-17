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
import numpy as np
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, Reshape, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
#Loading and reshaping the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_size = x_train.shape[1]

x_train = x_train.astype('float32') / 255.
x_train_final = x_train.reshape(len(x_train), img_size, img_size, 1)

x_test = x_test.astype('float32') / 255.
x_test_final = x_test.reshape(len(x_test), img_size, img_size, 1)

print(x_train_final.shape, x_test_final.shape)
#Adding noise to data
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train_final.shape)
x_train_noisy = x_train_final + noise

noise = np.random.normal(loc=0.5, scale=0.5, size=x_test_final.shape)
x_test_noisy = x_test_final + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
#Building the Denoising Autoencoder model
input_shape = (28, 28, 1)

inputs = Input(shape=input_shape, name = 'encoder_input')
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) #28*28*32
x = MaxPooling2D((2, 2), padding='same')(x) #14*14*32
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) #14*14*32
encoder = MaxPooling2D((2, 2), padding='same')(x) #7*7*32

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder) #7*7*32
x = UpSampling2D((2, 2))(x) #14*14*32
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) #14*14*32
x = UpSampling2D((2, 2))(x) #28*28*32
decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) #28*28*1

dae = Model(inputs, decoder)
dae.compile(optimizer='adam', loss='binary_crossentropy')

dae.summary()
dae.fit(x_train_noisy, x_train_final,
                    epochs=25, 
                    batch_size=256,
                    shuffle=True)
#Visualizing the outputs
decoded_imgs = dae.predict(x_test_noisy)

n = 10
plt.figure(figsize=(15, 4))
for i in range(n):
  
    # display original noisy image
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))


    # display denosined image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))

plt.show()
