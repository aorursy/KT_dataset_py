# Load packages

import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd

import tensorflow as tf



from skimage.io import imread

from skimage.transform import resize



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import AvgPool2D, Conv2D, MaxPool2D



DIR = '../input/the-simpsons-characters-dataset/simpsons_dataset/homer_simpson/'
# Load an image

sample_image = imread(f'{DIR}/pic_0042.jpg').astype('float32')
print(f'The shape of the image is {sample_image.shape}.')
# Plot the image

plt.imshow(sample_image.astype('uint8'))

plt.show()
# Create 2D-convolutional layer

conv = Conv2D(filters=3, kernel_size=(5, 5), padding='same', input_shape=(None, None, 3))
# Expand dimensions of the image

img_in = np.expand_dims(sample_image, 0)
print(f'The shape of the expanded image is {img_in.shape}.')
# Convolution on the image

img_out = conv(img_in)
print(f'The type of the output image is {type(img_out)}. The shape of the output image is {img_out.shape}.')
# Convert to numpy

img_out_np = img_out[0].numpy()

print(f'The type of the converted output image is {type(img_out_np)}.')
# Plot images

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))



ax0.imshow(sample_image.astype('uint8'))

ax0.set_title('Input')



ax1.imshow(img_out_np.astype('uint8'))

ax1.set_title('Output')



plt.show()
print(f'The number of trainable parameters is {conv.count_params()}.')
weights = conv.get_weights()[0]

biases = conv.get_weights()[1]
print(f'Shape of the weights: {weights.shape} / Shape of the biases: {biases.shape}.')
def kernel_init(shape=(5, 5, 3, 3), dtype=None):

    array = np.zeros(shape=shape, dtype='float32')

    array[:, :, 0, 0] = 1 / 25

    array[:, :, 1, 1] = 1 / 25

    array[:, :, 2, 2] = 1 / 25

    return array
# Create 2D-convolutional layer

conv = Conv2D(filters=3, kernel_size=(5, 5), padding='same', 

              input_shape=(None, None, 3), kernel_initializer=kernel_init)
# Plot images

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))



ax0.imshow(sample_image.astype('uint8'))

ax0.set_title('Input')



ax1.imshow(conv(img_in)[0].numpy().astype('uint8'))

ax1.set_title('Output')



plt.show()
def kernel_init(shape=(5, 5, 3, 3), dtype=None):

    array = np.zeros(shape=shape, dtype='float32')

    # array[2, 2] select only the center of the kernel

    array[2, 2] = np.eye(3)

    return array
# Create 2D-convolutional layer

conv = Conv2D(filters=3, kernel_size=(5, 5), padding='same', 

              input_shape=(None, None, 3), kernel_initializer=kernel_init)
# Plot images

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))



ax0.imshow(sample_image.astype('uint8'))

ax0.set_title('Input')



ax1.imshow(conv(img_in)[0].numpy().astype('uint8'))

ax1.set_title('Output')



plt.show()
print(f'The shape of the output image is {conv(img_in)[0].numpy().shape}.')
# Create 2D-convolutional layer

conv = Conv2D(filters=3, kernel_size=(5, 5), padding='same', strides=2,

              input_shape=(None, None, 3), kernel_initializer=kernel_init)
# Plot images

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))



ax0.imshow(sample_image.astype('uint8'))

ax0.set_title('Input')



ax1.imshow(conv(img_in)[0].numpy().astype('uint8'))

ax1.set_title('Output')



plt.show()
print(f'The shape of the output image is {conv(img_in)[0].numpy().shape}.')
# Create 2D-convolutional layer

conv = Conv2D(filters=3, kernel_size=(5, 5), padding='valid',

              input_shape=(None, None, 3), kernel_initializer=kernel_init)
# Plot images

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))



ax0.imshow(sample_image.astype('uint8'))

ax0.set_title('Input')



ax1.imshow(conv(img_in)[0].numpy().astype('uint8'))

ax1.set_title('Output')



plt.show()
print(f'The shape of the output image is {conv(img_in)[0].numpy().shape}.')
# Convert image to greyscale

sample_image_grey = sample_image.mean(axis=2)



# Add the channel dimension even if it's only one channel

# to be consistent with Keras expectations

sample_image_grey = sample_image_grey[:, :, np.newaxis]
# Plot the image

plt.imshow(np.squeeze(sample_image_grey.astype('uint8')), cmap=plt.cm.gray)

plt.show()
def kernel_init(shape=(3, 3, 1, 1), dtype=None):

    array = np.zeros(shape=shape, dtype='float32')

    array[:, :, 0, 0] = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / 12

    return array
img_grey_in = np.expand_dims(sample_image_grey, 0)
# Create 2D-convolutional layer

conv = Conv2D(filters=1, kernel_size=(3, 3), padding='same',

              input_shape=(None, None, 1), kernel_initializer=kernel_init)
# Plot images

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))



ax0.imshow(np.squeeze(sample_image_grey.astype('uint8')), cmap=plt.cm.gray)

ax0.set_title('Input')



ax1.imshow(np.squeeze(conv(img_grey_in)[0].numpy().astype('uint8')), cmap=plt.cm.gray)

ax1.set_title('Output')



plt.show()
max_conv = MaxPool2D(2, strides=2, input_shape=(None, None, 3))
# Plot images

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))



ax0.imshow(sample_image.astype('uint8'))

ax0.set_title('Input')



ax1.imshow(max_conv(img_in)[0].numpy().astype('uint8'))

ax1.set_title('Output')



plt.show()
avg_conv = AvgPool2D(3, strides=3, padding='same', input_shape=(None, None, 3))
# Plot images

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))



ax0.imshow(sample_image.astype('uint8'))

ax0.set_title('Input')



ax1.imshow(avg_conv(img_in)[0].numpy().astype('uint8'))

ax1.set_title('Output')



plt.show()
def kernel_init(shape=(3, 3, 3, 3), dtype=None):

    array = np.zeros(shape=shape, dtype='float32')

    array[:, :, 0, 0] = 1 / 9

    array[:, :, 1, 1] = 1 / 9

    array[:, :, 2, 2] = 1 / 9

    return array
# Create 2D-convolutional layer

conv_avg = Conv2D(filters=3, kernel_size=(3, 3), padding='valid', strides=3,

              input_shape=(None, None, 3), kernel_initializer=kernel_init)
# Plot images

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))



ax0.imshow(sample_image.astype('uint8'))

ax0.set_title('Input')



ax1.imshow(conv_avg(img_in)[0].numpy().astype('uint8'))

ax1.set_title('Output')



plt.show()