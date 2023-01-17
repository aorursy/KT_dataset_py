import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import h5py


import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage import img_as_uint



from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import keras
from keras import optimizers

%matplotlib inline

import tensorflow as tf
import sklearn
import skimage
from skimage import transform
from os import environ

seed = 42

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Skimage      :', skimage.__version__)
print('Scikit-learn :', sklearn.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
img_rows = 64
img_cols = 64

TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'
IMG_TYPE = '.png'         # Image type
IMG_DIR_NAME = 'images'   # Folder name including the image
MASK_DIR_NAME = 'masks'   # Folder name including the masks
LOGS_DIR_NAME = 'logs'    # Folder name for TensorBoard summaries 
SAVES_DIR_NAME = 'saves'  # Folder name for storing network parameters
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# Collection of methods for data operations. Implemented are functions to read  
# images/masks from files and to read basic properties of the train/test data sets.
import cv2

def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size: 
        img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
    return img

def read_mask(directory, target_size=None):
    """Read and resize masks contained in a given directory."""
    for i,filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, target_size)
        if not i: mask = mask_tmp
        else: mask = np.maximum(mask, mask_tmp)
    return mask 

def read_train_data_properties(train_dir, img_dir_name, mask_dir_name):
    """Read basic properties of training images and masks"""
    tmp = []
    for i,dir_name in enumerate(next(os.walk(train_dir))[1]):

        img_dir = os.path.join(train_dir, dir_name, img_dir_name)
        mask_dir = os.path.join(train_dir, dir_name, mask_dir_name) 
        num_masks = len(next(os.walk(mask_dir))[2])
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0]/img_shape[1], img_shape[2], num_masks,
                    img_path, mask_dir])

    train_df = pd.DataFrame(tmp, columns = ['img_id', 'img_height', 'img_width',
                                            'img_ratio', 'num_channels', 
                                            'num_masks', 'image_path', 'mask_dir'])
    return train_df

def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""
    tmp = []
    sizes_test = []
    for i,dir_name in enumerate(next(os.walk(test_dir))[1]):

        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        sizes_test.append([img_shape[0], img_shape[1]])

        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0]/img_shape[1], img_shape[2], img_path])

    test_df = pd.DataFrame(tmp, columns = ['img_id', 'img_height', 'img_width',
                                           'img_ratio', 'num_channels', 'image_path'])
    return test_df, sizes_test

def imshow_args(x):
    """Matplotlib imshow arguments for plotting."""
    if len(x.shape)==2: return x, cm.gray
    if x.shape[2]==1: return x[:,:,0], cm.gray
    return x, None

def load_raw_data(image_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Load raw data."""
    # Python lists to store the training images/masks and test images.
    x_train, y_train, x_test = [],[],[]

    # Read and resize train images/masks. 
    print('Loading and resizing train images and masks ...')
    sys.stdout.flush()
    for i, filename in tqdm(enumerate(train_df['image_path']), total=len(train_df)):
        #img = read_image(train_df['image_path'].loc[i], target_size=image_size)
        #mask = read_mask(train_df['mask_dir'].loc[i], target_size=image_size)
        img = read_image(train_df['image_path'].loc[i]) # dont resize
        mask = read_mask(train_df['mask_dir'].loc[i]) # dont resize
        
        x_train.append(img)
        y_train.append(mask)

    # Read and resize test images. 
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm(enumerate(test_df['image_path']), total=len(test_df)):
        #img = read_image(test_df['image_path'].loc[i], target_size=image_size)
        img = read_image(test_df['image_path'].loc[i]) #dont resize
        x_test.append(img)

    # Transform lists into 4-dim numpy arrays.
    x_train = np.array(x_train)
    #y_train = np.expand_dims(np.array(y_train), axis=4)
    y_train = np.array(y_train)
    x_test = np.array(x_test)

    #print('x_train.shape: {} of dtype {}'.format(x_train.shape, x_train.dtype))
    #print('y_train.shape: {} of dtype {}'.format(y_train.shape, x_train.dtype))
    #print('x_test.shape: {} of dtype {}'.format(x_test.shape, x_test.dtype))
    
    return x_train, y_train, x_test
# Basic properties of images/masks. 
train_df = read_train_data_properties(TRAIN_PATH, IMG_DIR_NAME, MASK_DIR_NAME)
test_df, sizes_test = read_test_data_properties(TEST_PATH, IMG_DIR_NAME)
print('train_df:')
print(train_df.describe())
print('')
print('test_df:')
print(test_df.describe())
print(sizes_test[0])
X_train, Y_train, X_test = load_raw_data()
print('X_Train={} , Y_train={}, X_test={}'.format(X_train.shape, Y_train.shape, X_test.shape))
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def form_batch(X, Y, batch_size):
    X_batch = np.zeros((batch_size, img_rows, img_cols, IMG_CHANNELS))
    Y_batch = np.zeros((batch_size, img_rows, img_cols, 1))
    

    for i in range(batch_size):
        #Every batch consists of images from multiple random image
        #Other way of doing it is to make eatch batch consists of patch from same image
        random_image_idx = np.random.randint(len(X))
        x = X[random_image_idx]
        y = Y[random_image_idx]
        X_height = x.shape[0]
        X_width = x.shape[1]

        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)
        
        Y_batch[i] = np.expand_dims(y[random_height: random_height + img_rows, random_width: random_width + img_cols]
                                   , axis=3)
        X_batch[i] = np.array(x[random_height: random_height + img_rows, 
                                random_width: random_width + img_cols, : ])
    return X_batch, Y_batch

def batch_generator(X, Y, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True):
    #count = 0
    while True:
        #count += 1
        X_batch, Y_batch = form_batch(X, Y, batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = Y_batch[i]

            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 2)
                    yb = flip_axis(yb, 2)

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(0, 1)
                    yb = yb.swapaxes(0, 1)

            X_batch[i] = xb
            Y_batch[i] = yb
        #change this to yield when using as generator
        return X_batch, Y_batch
batch_size = 16
no_cols = 4
X_batch, Y_batch = batch_generator(X_train, Y_train, batch_size)
plt.close('all')
no_rows = int(batch_size/no_cols)

fig, axes = plt.subplots(no_rows*2, no_cols, figsize = (20, 20))

count = 0
for i in range(no_rows):
    for j in range(no_cols):
        axes[2*i, j].imshow(X_batch[count])
        axes[2*i, j].axis('off')

        axes[2*i+1, j].imshow(np.squeeze(Y_batch[count]))
        axes[2*i+1, j].axis('off')
        axes[2*i+1, j].set_title('{}{} th True Y patch'.format(i,j))
        count +=1

