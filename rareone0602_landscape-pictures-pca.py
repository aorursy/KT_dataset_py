import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow_probability as tfp

import tensorflow as tf

import os

import random



img_paths = sorted([os.path.join(dirname, filename) for dirname, _, filenames in os.walk('/kaggle/input/landscape-pictures') for filename in filenames])
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 64

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width), color_mode="grayscale") for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    return img_array

def flatten(imag):

    vec = np.reshape(imag, (-1, image_size * image_size))

    return vec

def unflatten(vec):

    imag = np.reshape(vec, (-1, image_size, image_size))

    return imag
raw_data = np.reshape(read_and_prep_images(img_paths), (-1, image_size, image_size))
import matplotlib.pyplot as plt

%matplotlib inline

plt.imshow(raw_data[1]/256, cmap='gray')

plt.show()
flattened = flatten(raw_data)

N = len(flattened)

mean = np.mean(flattened, axis=0)

X = flattened - mean
plt.imshow(np.reshape(mean, (image_size, image_size))/256, cmap='gray')

plt.show()
from sklearn import decomposition

pca = decomposition.PCA()

pca.fit(X)

ys = pca.transform(X)
w = 20

h = 20

fig = plt.figure(figsize = (w, h))

columns = 8

rows = 8

for i in range(64, 4096 + 1, 64):

    fig.add_subplot(rows, columns, i//64)

    decoded = np.matmul(ys[1,:i], pca.components_[:i, :]) + mean

    plt.imshow(np.reshape(decoded, (image_size, image_size)), cmap='gray')

plt.show()
print(np.max(np.absolute(np.matmul(ys, pca.components_) - X))/np.max(np.absolute(X)))
plt.plot(pca.explained_variance_)

plt.show()
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.show()
plt.yscale(value='log')

plt.plot(pca.explained_variance_)

plt.show()
components = unflatten(np.multiply(-pca.components_, 2 * np.sqrt(pca.explained_variance_)[:,np.newaxis]) + mean)
w = 20

h = 20

fig = plt.figure(figsize = (w, h))

columns = 6

rows = 6

for i in range(columns * rows):

    fig.add_subplot(rows, columns, i + 1)

    plt.imshow(components[i], cmap='gray')

plt.show()
components = unflatten(np.multiply(pca.components_, 2 * np.sqrt(pca.explained_variance_)[:,np.newaxis]) + mean)
w = 20

h = 20

fig = plt.figure(figsize = (w, h))

columns = 6

rows = 6

for i in range(columns * rows):

    fig.add_subplot(rows, columns, i + 1)

    plt.imshow(components[i], cmap='gray')

plt.show()