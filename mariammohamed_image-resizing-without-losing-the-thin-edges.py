import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2

import keras

from skimage.transform import resize

from keras.layers import *

from keras.models import *

from keras.optimizers import *
train_data = pd.read_csv('../input/train.csv')
unique_images = train_data.ImageId.unique()
def get_masks(image_id, resized_shape=(128, 128)):

  masks = dict()

  temp = train_data[train_data.ImageId == image_id]

  for i in range(temp.shape[0]):

    width = temp.iloc[i].Width

    height = temp.iloc[i].Height

    class_id = temp.iloc[i].ClassId.split()[0]

    mask_encoded = temp.iloc[i].EncodedPixels.split()

    mask = [0] * (width*height)

    for j in range(0, len(mask_encoded), 2):

      mask[int(mask_encoded[j]): int(mask_encoded[j])+int(mask_encoded[j+1])] = [1]*int(mask_encoded[j+1])

    mask = np.fliplr(np.flip(np.rot90(np.array(mask).reshape((width, height)))))

    mask = resize(mask, resized_shape, anti_aliasing=True)

    masks[int(class_id)] = mask

  masks_classes = []

  for i in range(46):

    if i in masks:

      masks_classes.append(masks[i])

    else:

      masks_classes.append(np.zeros(resized_shape))

  masks_classes = np.array(masks_classes)

  return masks_classes
train_data[train_data.ImageId == unique_images[1]]
masks = get_masks(unique_images[1], (2448, 2448))
plt.imshow(masks[33])
masks = get_masks(unique_images[1], (128, 128))
plt.imshow(masks[33])
model_resize = Sequential()

model_resize.add(MaxPool2D((2, 2), data_format='channels_first', input_shape=(46, 1024, 1024)))

model_resize.add(MaxPool2D((2, 2), data_format='channels_first'))

model_resize.add(MaxPool2D((2, 2), data_format='channels_first'))
masks = get_masks(unique_images[1], (1024, 1024))

masks = model_resize.predict(np.array([masks]))[0]
plt.imshow(masks[33])