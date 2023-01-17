%matplotlib inline

import numpy as np

import cv2

import matplotlib.pyplot as plt

import os
plt.style.use('ggplot')
def imshow(img):

    plt.imshow(img, cmap='gray')
img = cv2.imread('../input/test/1.png', 0)

imshow(img)
def generate_data():

    test_imgs = []

    train_imgs = []

    

    files = os.listdir('../input/train_cleaned/')

    for filename in files:

        img = cv2.imread('../input/train_cleaned/'+filename, 0)

        img = cv2.resize(img, (540, 258))

        train_imgs.append(img)



    train_imgs = np.array(train_imgs)

    

    files = os.listdir('../input/test/')

    for filename in files:

        img = cv2.imread('../input/test/'+filename, 0)

        img = cv2.resize(img, (540, 258))

        test_imgs.append(img)



    test_imgs = np.array(test_imgs)

    

    return (train_imgs, test_imgs)
def generate_validation_set(train_set, test_size=0.4):

    cut_off = int(train_set.shape[0] * test_size)

    X_test = train[ : cut_off, :, :]

    X_train = train[cut_off : , :, :]

    

    return (X_train, X_test)
train, test = generate_data()
X_train, X_test = generate_validation_set(train)
imshow(X_train[0])
X_train = X_train.astype('float32') / 255.

X_test = X_test.astype('float32') / 255.

test = test.astype('float32') / 255.
X_train = X_train.reshape(-1, (X_train.shape[1] * X_train.shape[2]))

X_test = X_test.reshape(-1, (X_test.shape[1] * X_test.shape[2]))

test = test.reshape(-1, (test.shape[1] * test.shape[2]))
X_train.shape
from keras.layers import Input, Dense

from keras.models import Model

from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
inputs = Input(shape=(1, 258, 540))

x = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputs)

x = MaxPooling2D(2, 2, border_mode='same')(x)

x = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(x)

encoded = MaxPooling2D(2, 2, border_mode='same')(x)



x = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(encoded)

x = UpSampling2D(2, 2, border_mode='same')(x)

x = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(x)

x = UpSampling2D(2, 2, border_mode='same')(x)

decoded = Convolution2D(1, 3, 3, border_mode='same', activation='relu')(x)
np.random.seed(2016)
decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs.shape
decoded_imgs = decoded_imgs.reshape((-1, 258, 540))
imshow(decoded_imgs[1, :, :])
from sklearn.cluster import KMeans
img = X_train[0, :]

imshow(img.reshape(258, 540))