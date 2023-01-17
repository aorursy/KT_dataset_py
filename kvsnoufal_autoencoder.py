# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# LOAD LIBRARIES

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,InputLayer,Conv2DTranspose

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt



%matplotlib inline



import numpy as np

import keras

from keras.datasets import mnist

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from keras import regularizers
# LOAD THE DATA

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

X_train = X_train / 255.0

X_test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

X_train.shape,X_test.shape
def train_val_split(x_train, y_train):

    rnd = np.random.RandomState(seed=42)

    perm = rnd.permutation(len(x_train))

    train_idx = perm[:int(0.8 * len(x_train))]

    val_idx = perm[int(0.8 * len(x_train)):]

    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]



n = 20000  # for 2 random indices

index = np.random.choice(X_train.shape[0], n, replace=False) 

X_train=X_train[index]

x_train, y_train, x_val, y_val = train_val_split(X_train, X_train)



max_value = float(x_train.max())

x_train = x_train.astype('float32') / max_value

x_val = x_val.astype('float32') / max_value

print(x_train.shape, x_val.shape)


input=Input(shape=(x_train.shape[1:]))

encoded=Conv2D(16, (3, 3), activation='relu', padding='same')(input)

encoded=MaxPooling2D((2, 2), padding='same')(encoded)

encoded=Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)

encoded=MaxPooling2D((2, 2), padding='same')(encoded)

encoded=Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(encoded)

encoded=Flatten()(encoded)



decoded=Reshape((4, 4, 8))(encoded)

decoded=Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)

decoded=UpSampling2D((2, 2))(decoded)

decoded=Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)

decoded=UpSampling2D((2, 2))(decoded)

decoded=Conv2D(16, (3, 3), activation='relu')(decoded)

decoded=UpSampling2D((2, 2))(decoded)

decoded=Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)

        

autoencoder=Model(input,decoded)





autoencoder.summary()
x_train.shape
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[6].output)

encoder.summary()
encoded_input = Input(shape=(128,))



deco = autoencoder.layers[-8](encoded_input)

deco = autoencoder.layers[-7](deco)

deco = autoencoder.layers[-6](deco)

deco = autoencoder.layers[-5](deco)

deco = autoencoder.layers[-4](deco)

deco = autoencoder.layers[-3](deco)

deco = autoencoder.layers[-2](deco)

deco = autoencoder.layers[-1](deco)

# create the decoder model

decoder = Model(encoded_input, deco)

decoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
len(autoencoder.layers),len(encoder.layers),len(decoder.layers)
from keras.preprocessing import image

gen = image.ImageDataGenerator()

batches = gen.flow(x_train, x_train, batch_size=64)

val_batches=gen.flow(x_val, x_val, batch_size=64)
history=autoencoder.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=4, 

                    validation_data=val_batches, validation_steps=val_batches.n)
test_image=x_val[0].reshape(1,28,28,1)

plt.imshow(test_image.reshape(28, 28))

plt.show()

encoded_img=encoder.predict(test_image)

plt.imshow(encoded_img)

plt.show()

decoded_img=decoder.predict(encoded_img)

plt.imshow(decoded_img.reshape(28,28))

plt.show()

decoded_img2=autoencoder.predict(test_image)

plt.imshow(decoded_img2.reshape(28,28))

plt.show()
test_image=x_val[2].reshape(1,28,28,1)

plt.imshow(test_image.reshape(28, 28))

plt.show()

encoded_img=encoder.predict(test_image)

plt.imshow(encoded_img)

plt.show()

decoded_img=decoder.predict(encoded_img)

plt.imshow(decoded_img.reshape(28,28))

plt.show()

decoded_img2=autoencoder.predict(test_image)

plt.imshow(decoded_img2.reshape(28,28))

plt.show()
import numpy as np

def interpolate_points(p1, p2, n_steps=10):

	# interpolate ratios between the points

	ratios = np.linspace(0, 1, num=n_steps)

	# linear interpolate vectors

	vectors = list()

	for ratio in ratios:

		v = (1.0 - ratio) * p1 + ratio * p2

		vectors.append(v)

	return np.asarray(vectors)
# example of interpolating 

from numpy import asarray

from numpy import vstack

from numpy.random import randn

from numpy.random import randint

from numpy import arccos

from numpy import clip

from numpy import dot

from numpy import sin

from numpy import linspace

from numpy.linalg import norm

from keras.models import load_model

# # spherical linear interpolation (slerp)

# def slerp(val, low, high):

# 	omega = arccos(clip(dot(low/norm(low), high/norm(high)), -1, 1))

# 	so = sin(omega)

# 	if so == 0:

# 		# L'Hopital's rule/LERP

# 		return (1.0-val) * low + val * high

# 	return sin((1.0-val)*omega) / so * low + sin(val*omega) / so * high



# uniform interpolation between two points in latent space

def interpolate_points_slerp(p1, p2, n_steps=10):

	# interpolate ratios between the points

	ratios = linspace(0, 1, num=n_steps)

	# linear interpolate vectors

	vectors = list()

	for ratio in ratios:

		v = slerp(ratio, p1, p2)

		vectors.append(v)

	return asarray(vectors)
test_image1=x_val[0].reshape(1,28,28,1)

test_image2=x_val[2].reshape(1,28,28,1)

encoded_img1=encoder.predict(test_image1)

encoded_img2=encoder.predict(test_image2)

interpolated_images=interpolate_points(encoded_img1.flatten(),encoded_img2.flatten())

interpolated_orig_images=interpolate_points(test_image1.flatten(),test_image2.flatten())

# interpolated_slerp_images=interpolate_points_slerp(encoded_img1.flatten(),encoded_img2.flatten())

# interpolated_slerp_orig_images=interpolate_points_slerp(test_image1.flatten(),test_image2.flatten())



interpolated_images.shape

num_images = 10

np.random.seed(42)

plt.figure(figsize=(20, 8))



for i, image_idx in enumerate(interpolated_images):

    

    ax = plt.subplot(5, num_images, i + 1)

    plt.imshow(interpolated_images[i].reshape(16, 8))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    ax.set_title("Encoded: {}".format(i))

    

    ax = plt.subplot(5, num_images,num_images+ i + 1)

    plt.imshow(decoder.predict(interpolated_images[i].reshape(1,128)).reshape(28,28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    ax.set_title("Latent: {}".format(i))

    

#     ax.get_yaxis().set_visible(False)

#     ax = plt.subplot(5, num_images,2*num_images+ i + 1)

#     plt.imshow(decoder.predict(interpolated_slerp_images[i].reshape(1,128)).reshape(28,28))

#     plt.gray()

#     ax.get_xaxis().set_visible(False)

#     ax.get_yaxis().set_visible(False)

    

    ax = plt.subplot(5, num_images,2*num_images+ i + 1)

    plt.imshow(interpolated_orig_images[i].reshape(28,28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    ax.set_title("Image: {}".format(i))

    

#     ax = plt.subplot(5, num_images,4*num_images+ i + 1)

#     plt.imshow(interpolated_slerp_orig_images[i].reshape(28,28))

#     plt.gray()

#     ax.get_xaxis().set_visible(False)

#     ax.get_yaxis().set_visible(False)

plt.show()
interpolated_images.shape
encoder.save('encoder.h5')

decoder.save('decoder.h5')
print("Done")
# while True:

#     pass