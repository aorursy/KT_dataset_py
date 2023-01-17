# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/skin-cancer-malignant-vs-benign'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from glob import glob 

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2grey
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
mal_images = glob('../input/skin-cancer-malignant-vs-benign/data/train/malignant/*')
ben_images = glob('../input/skin-cancer-malignant-vs-benign/data/train/benign/*')
def load_images(paths):
    tmp = []
    for path in paths:
        tmp.append(imread(path))
    return tmp
mal = load_images(mal_images)
ben = load_images(ben_images)

from skimage import filters, segmentation

# find a dividing line between 0 and 255
# pixels below this value will be black
# pixels above this value will be white
for im in mal : 
    
    val = filters.threshold_otsu(im)
    mask = im < val
    clean_border = segmentation.clear_border(mask)
    plt.imshow(clean_border, cmap='gray')
    plt.show()

# the mask object converts each pixel in the image to True or False
# to indicate whether the given pixel is black/white


# apply the mask to the image object


# plot the resulting binarized image
