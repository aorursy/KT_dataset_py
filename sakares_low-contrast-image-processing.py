# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import glob



import pydicom

from skimage import data, img_as_float, exposure, io



print(os.listdir("../input/siim-acr-pneumothorax-segmentation/sample images"))

# Any results you write to the current directory are saved as output.



from matplotlib import cm

from matplotlib import pyplot as plt



import sys

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')
start = 0   # Starting index of images

num_img = 5 # Total number of images to show



fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')[start:start+num_img]):

    dataset = pydicom.dcmread(file_path)

    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.gray)

start = 0   # Starting index of images

num_img = 5 # Total number of images to show



fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')[start:start+num_img]):

    dataset = pydicom.dcmread(file_path)



    src_img = dataset.pixel_array

    

    # Contrast Stretching

    p2, p98 = np.percentile(src_img, (2, 98))

    img_cs = img_as_float(exposure.rescale_intensity(src_img, in_range=(p2, p98)))

    

    ax[q].imshow(img_cs, cmap=plt.cm.gray)
start = 0   # Starting index of images

num_img = 5 # Total number of images to show



fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')[start:start+num_img]):

    dataset = pydicom.dcmread(file_path)



    src_img = dataset.pixel_array

    

    # Histogram Equalization

    img_he = img_as_float(exposure.equalize_hist(src_img))

    

    ax[q].imshow(img_he, cmap=plt.cm.gray)
start = 0   # Starting index of images

num_img = 5 # Total number of images to show



fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')[start:start+num_img]):

    dataset = pydicom.dcmread(file_path)



    src_img = dataset.pixel_array

    

    # Adaptive Equalization

    img_ae = img_as_float(exposure.equalize_adapthist(src_img))

    

    ax[q].imshow(img_ae, cmap=plt.cm.gray)
def plot_img_and_hist(img, axes, bins=256):

    """Plot an image along with its histogram and cumulative histogram.



    """

    img = img_as_float(img)

    ax_img, ax_hist = axes

    ax_cdf = ax_hist.twinx()



    # Display image

    ax_img.imshow(img, cmap=plt.cm.gray)

    ax_img.set_axis_off()



    # Display histogram

    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')

    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    ax_hist.set_xlabel('Pixel intensity')

    ax_hist.set_xlim(0, 1)

    ax_hist.set_yticks([])



    # Display cumulative distribution

    img_cdf, bins = exposure.cumulative_distribution(img, bins)

    ax_cdf.plot(bins, img_cdf, 'r')

    ax_cdf.set_yticks([])



    return ax_img, ax_hist, ax_cdf
#### Everything below here is just to create the plot/graphs ####

# Display results

fig = plt.figure(figsize=(16, 10)) 

axes = np.zeros((2, 4), dtype=np.object)

axes[0, 0] = fig.add_subplot(2, 4, 1)



for i in range(1, 4):

    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])

    

for i in range(0, 4):

    axes[1, i] = fig.add_subplot(2, 4, 5+i)

    

sample_file_path = '../input/siim-acr-pneumothorax-segmentation/sample images/1.2.276.0.7230010.3.1.4.8323329.4982.1517875185.837576.dcm'

dataset = pydicom.dcmread(sample_file_path)



src_img = dataset.pixel_array

    

# Contrast Stretching

p2, p98 = np.percentile(src_img, (2, 98))

img_cs = img_as_float(exposure.rescale_intensity(src_img, in_range=(p2, p98)))

    

# Histogram Equalization

img_he = img_as_float(exposure.equalize_hist(src_img))

    

# Adaptive Equalization

img_ae = img_as_float(exposure.equalize_adapthist(src_img))

    

ax_img, ax_hist, ax_cdf = plot_img_and_hist(src_img, axes[:, 0])

ax_img.set_title('Low contrast image')



y_min, y_max = ax_hist.get_ylim()

ax_hist.set_ylabel('Number of pixels')



ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_cs, axes[:, 1])

ax_img.set_title('Contrast stretching')



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_he, axes[:, 2])

ax_img.set_title('Histogram equalization')



ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_ae, axes[:, 3])

ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')

ax_cdf.set_yticks(np.linspace(0, 1, 5))



# prevent overlap of y-axis labels

fig.tight_layout()

plt.show()