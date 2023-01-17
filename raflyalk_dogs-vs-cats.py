import os

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.image import imread

from os import listdir

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array



print(os.listdir('../input/dogs-vs-cats/'))
# define location of dataset

folder = '../input/dogs-vs-cats/train/train/'

# plot first few images

for i in range(9):

	# define subplot

	plt.subplot(330 + 1 + i)

	# define filename

	filename = folder + 'dog.' + str(i) + '.jpg'

	# load image pixels

	image = imread(filename)

	# plot raw pixel data

	plt.imshow(image)

# show the figure

plt.show()
# define location of dataset

folder = '../input/dogs-vs-cats/train/train/'

# plot first few images

for i in range(9):

	# define subplot

	plt.subplot(330 + 1 + i)

	# define filename

	filename = folder + 'cat.' + str(i) + '.jpg'

	# load image pixels

	image = imread(filename)

	# plot raw pixel data

	plt.imshow(image)

# show the figure

plt.show()
# define location of dataset

folder = '../input/dogs-vs-cats/train/train/'

photos, labels = list(), list()

# enumerate files in the directory

for file in listdir(folder):

	# determine class

	output = 0.0

	if file.startswith('cat'):

		output = 1.0

	# load image

	photo = load_img(folder + file, target_size=(100, 100))

	# convert to numpy array

	photo = img_to_array(photo)

	# store

	photos.append(photo)

	labels.append(output)

# convert to a numpy arrays

photos = np.asarray(photos)

labels = np.asarray(labels)

print(photos.shape, labels.shape)
np.save('dogs_vs_cats_photos.npy', photos)

np.save('dogs_vs_cats_labels.npy', labels)
from numpy import load

photos = load('dogs_vs_cats_photos.npy')

labels = load('dogs_vs_cats_labels.npy')

print(photos.shape, labels.shape)