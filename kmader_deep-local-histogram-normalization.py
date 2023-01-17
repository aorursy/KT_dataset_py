import numpy as np # for manipulating 3d images

import pandas as pd # for reading and writing tables

import h5py # for reading the image files

import skimage # for image processing and visualizations

import sklearn # for machine learning and statistical models

import os # help us load files and deal with paths
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['image.cmap'] = 'gray'

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
train_image = skimage.io.imread('../input/training.tif')

print(train_image.shape)

sample_slice = np.expand_dims(np.expand_dims(train_image[train_image.shape[0]//2], 0), -1)

plt.imshow(sample_slice[0, :, :, 0], cmap='gray')
from keras import layers, models, optimizers, losses, backend as K
PATCH_SIZE = (32, 32)
def build_dlhn(patch_size = PATCH_SIZE, patch_strides = None, clip_max=1/255.0):

    if patch_strides is None:

        patch_strides = (np.clip(patch_size[0]//2, 1, patch_size[0]), 

                         np.clip(patch_size[1]//2, 1, patch_size[1]))

    image_in = layers.Input((None, None, 1))

    flip_layer = layers.Conv2D(1, (1, 1), use_bias=False, activation='linear', weights=[-1*np.ones((1, 1, 1, 1))], name='FlipLayer')

    flip_layer.trainable = False

    region_max_image = layers.MaxPool2D(patch_size, strides=patch_strides, name='MaxLayer', padding='same')(image_in)

    region_min_image = flip_layer(layers.MaxPool2D(patch_size, strides=patch_strides, name='MinLayer', padding='same')(flip_layer(image_in)))

    if np.max(patch_strides)>1:

        region_max_image = layers.UpSampling2D(patch_strides, name='UpsampleMax')(region_max_image)

        region_min_image = layers.UpSampling2D(patch_strides, name='UpsampleMin')(region_min_image)

    region_diff_image = layers.subtract([region_max_image, region_min_image], name='CalculateRange')

    region_min_shift = layers.subtract([image_in, region_min_image], name='RemoveOffset')

    norm_image = layers.Lambda(lambda x: x[0]/K.clip(x[1], clip_max, 1/clip_max), name='DivideByRange')([region_min_shift, region_diff_image])

    return models.Model(inputs=[image_in], outputs=[norm_image])

dlhn_model = build_dlhn(patch_size=PATCH_SIZE)

dlhn_model.summary()
from keras.utils import vis_utils

from IPython.display import SVG

SVG(vis_utils.model_to_dot(dlhn_model, show_shapes=True).create_svg())
fig, m_axs = plt.subplots(2, 3, figsize=(12, 8))

for (ax1, ax2), c_img in zip(m_axs.T, 

                             [sample_slice[0, :, :, 0], 

                              dlhn_model.predict(sample_slice)[0, :, :, 0], 

                              skimage.exposure.equalize_adapthist(sample_slice[0, :, :, 0], kernel_size=PATCH_SIZE)]):

    ax1.imshow(c_img, cmap='gray')

    ax2.hist(c_img.ravel())

%%time

_ = dlhn_model.predict(np.expand_dims(train_image, -1), verbose=True, batch_size=32)
%%time

_ = [skimage.exposure.equalize_adapthist(c_slice, kernel_size=PATCH_SIZE) for c_slice in train_image]