import sys

import os

import subprocess

from six import string_types



# Make sure you have all of these packages installed, e.g. via pip

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import scipy

from skimage import io

from scipy import ndimage

from IPython.display import display

%matplotlib inline
PLANET_KAGGLE_ROOT = os.path.abspath("../input/")

PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')

PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train.csv')

assert os.path.exists(PLANET_KAGGLE_ROOT)

assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)

assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)
def load_image(filename):

    for dirname in os.listdir(PLANET_KAGGLE_ROOT):

        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))

        if os.path.exists(path):

            print('Found image {}'.format(path))

            return io.imread(path)

    # if you reach this line, you didn't find the image you're looking for

    print('Load failed: could not find image {}'.format(path))
def calibrate_image(rgb_image):

    # Transform test image to 32-bit floats to avoid 

    # surprises when doing arithmetic with it 

    calibrated_img = rgb_image.copy().astype('float32')



    # Loop over RGB

    for i in range(3):

        # Subtract mean 

        calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.mean(calibrated_img[:,:,i])

        # Normalize variance

        calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.std(calibrated_img[:,:,i])

        # Scale to reference 

        calibrated_img[:,:,i] = calibrated_img[:,:,i]*ref_stds[i] + ref_means[i]

        # Clip any values going out of the valid range

        calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)



    # Convert to 8-bit unsigned int

    return calibrated_img.astype('uint8')
# Pull a list of 20000 image names

jpg_list = os.listdir(PLANET_KAGGLE_JPEG_DIR)[:20000]

# Select a random sample of 100 among those

np.random.shuffle(jpg_list)

jpg_list = jpg_list[:100]
ref_colors = [[],[],[]]

for _file in jpg_list:

    # keep only the first 3 bands, RGB

    _img = mpimg.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]

    # Flatten 2-D to 1-D

    _data = _img.reshape((-1,3))

    # Dump pixel values to aggregation buckets

    for i in range(3): 

        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()

    

ref_colors = np.array(ref_colors)
ref_means = [np.mean(ref_colors[i]) for i in range(3)]

ref_stds = [np.std(ref_colors[i]) for i in range(3)]
def show_img(path):

    img = load_image(path)[:,:,:3]

    if '.tif' in path:

        img = calibrate_image(img)

    fig = plt.figure()

    a = fig.add_subplot(1, 1, 1)

    a = a.set_title(path)

    plt.imshow(img)
for i in range(1000, 1010):

    show_img('train_{}.tif'.format(i))
show_img('train_10684.tif')
show_img('train_1000.tif')
show_img('train_1.tif')
show_img('train_5023.tif')
def sample_images(tags, n=None):

    """Randomly sample n images with the specified tags."""

    condition = True

    if isinstance(tags, string_types):

        raise ValueError("Pass a list of tags, not a single tag.")

    for tag in tags:

        condition = condition & labels_df[tag] == 1

    if n is not None:

        return labels_df[condition].sample(n)

    else:

        return labels_df[condition]
def load_image(filename):

    '''Look through the directory tree to find the image you specified

    (e.g. train_10.tif vs. train_10.jpg)'''

    for dirname in os.listdir(PLANET_KAGGLE_ROOT):

        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))

        if os.path.exists(path):

            print('Found image {}'.format(path))

            return io.imread(path)

    # if you reach this line, you didn't find the image you're looking for

    print('Load failed: could not find image {}'.format(path))

    

def sample_to_fname(sample_df, row_idx, suffix='tif'):

    '''Given a dataframe of sampled images, get the

    corresponding filename.'''

    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')

    return '{}.{}'.format(fname, suffix)
labels = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
def show_sample(n=10):

    sample = labels.sample(n)

    for fname in sample.image_name:

        fname = '{}.tif'.format(fname)

        img = show_img(fname)
show_sample(10)
show_sample(10)
show_sample(10)
show_sample(10)
show_sample(10)
show_sample(10)
show_sample(10)
show_sample(10)
show_sample(10)