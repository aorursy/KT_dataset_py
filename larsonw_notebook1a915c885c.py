# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import keras
from keras.layers import Input, Dense

from keras.models import Model



# this is the size of our encoded representations

encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats



# this is our input placeholder

input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input

encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input

decoded = Dense(784, activation='sigmoid')(encoded)



# this model maps an input to its reconstruction

autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input

encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model

decoder_layer = autoencoder.layers[-1]

# create the decoder model

decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist

import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.

x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)

print(x_test.shape)

autoencoder.fit(x_train, x_train,

                nb_epoch=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)

decoded_imgs = decoder.predict(encoded_imgs)
!ls
