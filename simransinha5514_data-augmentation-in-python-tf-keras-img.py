# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import skimage

from skimage import transform

from skimage import util

import cv2

import tensorflow as tf

from math import pi



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

import os

print(os.listdir("../input"))

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_train.head()
X_train = df_train.iloc[:, 1:]

Y_train = df_train.iloc[:, 0]
X_train.head()
Y_train.head()
X_train = np.array(X_train)

Y_train = np.array(Y_train)
def plot_digits(X, Y, shape):

    for i in range(9):

        plt.subplot(3, 3, i+1)

        plt.tight_layout()

        plt.imshow(X[i].reshape((28,28)), interpolation='none', cmap='gray')

        plt.title('Digit:{}'.format(Y[i]))

        plt.xticks([])

        plt.yticks([])

    plt.show()
plot_digits(X_train, Y_train, 28)
def rotate_image(X, degrees):

    X_flip = []

    for i in range(9):

        img = X[i].reshape((28, 28))

        img = skimage.transform.rotate(img, degrees)

        X_flip.append(img.reshape((784)))

    X_trfr = np.array(X_flip)

    X = np.concatenate((X, X_flip))

    return X
X_rot = rotate_image(X_train, -30)
plot_digits(X_rot[42000:], Y_train, 28)
def flip_digits(X):

    X_flip = []

    for i in range(9):

        img = X[i].reshape((28, 28))

        img = np.fliplr(img)

        X_flip.append(img.reshape((784)))

    X_trfr = np.array(X_flip)

    X = np.concatenate((X, X_flip))

    return X
X_flip = flip_digits(X_train)
plot_digits(X_flip[42000:], Y_train, 28)
def noise_image(X):

    X_flip = []

    for i in range(9):

        img = X[i].reshape((28, 28))

        img = skimage.util.random_noise(img, mode='pepper')

        X_flip.append(img.reshape((784)))

    X_trfr = np.array(X_flip)

    X = np.concatenate((X, X_flip))

    return X
X_noise = noise_image(X_train)
plot_digits(X_noise[42000:], Y_train, 28)
def scale_up_image(X, scale):

    X_flip = []

    for i in range(9):

        img = X[i].reshape((28, 28))

        img = skimage.transform.rescale(img, scale, clip=True)

        img = skimage.util.crop(img, ((0, 28), (0, 28)))

        X_flip.append(img.reshape((784)))

    X_trfr = np.array(X_flip)

    X = np.concatenate((X, X_flip))

    return X
X_scale = scale_up_image(X_train, 2)
plot_digits(X_scale[42000:], Y_train, 28)
def translate_image(X, h, w):

    X_flip = []

    M = np.float32([[1, 0, h], [0, 1, w]])

    for i in range(9):

        img = X[i].reshape((28, 28))

        img = img.astype(np.float32)

        img = cv2.warpAffine(img, M, (28, 28))

        X_flip.append(img.reshape((784)))

    X_trfr = np.array(X_flip)

    X = np.concatenate((X, X_flip))

    return X
X_translate = translate_image(X_train, 5, 5)
plot_digits(X_translate[42000:], Y_train, 28)
def flip_image_tf(X, mode):

    X_img = tf.placeholder(dtype=tf.float32, shape=(28, 28, 1), name='X')

    if mode == 1:

        tf_flip = tf.image.flip_left_right(X_img)

    elif mode == 2:

        tf_flip = tf.image.flip_up_down(X_img)

    elif mode == 3:

        tf_flip = tf.image.transpose_image(X_img)



    tf.global_variables_initializer()

    sess = tf.Session()

    X_flip = []

    for i in range(9):

        img = X[i].reshape((28, 28, 1))

        img_flip = sess.run([tf_flip], feed_dict={X_img:img})

        X_flip.append(img_flip[0].reshape((784)))

    return X_flip
X_lr = flip_image_tf(X_train, 1)

X_lr = np.array(X_lr)
plot_digits(X_lr, Y_train, 28)
X_ud = flip_image_tf(X_train, 2)

X_ud = np.array(X_ud)
plot_digits(X_ud, Y_train, 28)
X_tr = flip_image_tf(X_train, 3)

X_tr = np.array(X_tr)
plot_digits(X_tr, Y_train, 28)
def rotate_image_tf(X, rot_angle):

    X_img = tf.placeholder(dtype=tf.float32, shape=(28, 28, 1), name='X')

    angle = tf.placeholder(dtype=tf.float32, shape=(1), name='angle')

    tf_rot = tf.contrib.image.rotate(X_img, angle)

    tf.global_variables_initializer()

    sess = tf.Session()

    X_rot = []

    for i in range(9):

        img = X[i].reshape((28, 28, 1))

        rad = [rot_angle*pi/180]

        img_rot = sess.run([tf_rot], feed_dict={X_img:img, angle:rad})

        img_rot = img_rot[0].reshape((784))

        X_rot.append(img_rot)

    return X_rot
X_rot = rotate_image_tf(X_train, 45)

X_rot = np.array(X_rot)
plot_digits(X_rot, Y_train, 28)
from keras.preprocessing.image import ImageDataGenerator
def image_aug_keras(X):

    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, \

                                 width_shift_range=0.1, height_shift_range=0.1)

    X_aug = []

    for i in range(9):

        X_train2, Y_train2 = datagen.flow(X_train[i,:].reshape((1, 28, 28, 1)), \

                                          Y_train[i].reshape((1, 1, 1, 1))).next()

        X_aug.append(X_train2.reshape((28,28)))

    return X_aug
X_aug = image_aug_keras(X_train)

X_aug = np.array(X_aug)
plot_digits(X_aug, Y_train, 28)
from imgaug import augmenters as iaa
def flip_image_iaa(X):

    X_flip = []

    seq = iaa.Sequential([

        iaa.Fliplr(1)

    ])

    for i in range(9):

        img = X[i].reshape((1, 28, 28))

        img_aug = seq.augment_images(img)

        X_flip.append(img_aug[0].reshape((784)))

    return X_flip
X_flip = flip_image_iaa(X_train)

X_flip = np.array(X_flip)
plot_digits(X_flip, Y_train, 28)
def augment_image_iaa(X):

    X_rotate = []

    seq = iaa.Sequential([

        iaa.Affine(

            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},

            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},

            rotate=(-25, 25),

            shear=(-8, 8)

        )

    ])

    for i in range(9):

        img = X[i].reshape((1, 28, 28))

        img_aug = seq.augment_images(img)

        X_rotate.append(img_aug.reshape((784)))

    return X_rotate
X_aug = augment_image_iaa(X_train)

X_aug = np.array(X_aug)
plot_digits(X_aug, Y_train, 28)
X_aug = augment_image_iaa(X_train)

X_aug = np.array(X_aug)
plot_digits(X_aug, Y_train, 28)