import tensorflow as tf
# These are all the modules we'll be using later. Make sure you can import them

# before proceeding further.

%matplotlib inline

from __future__ import print_function

import matplotlib.pyplot as plt

import numpy as np

import os

import sys

import tarfile

from IPython.display import display, Image

from scipy import ndimage

from sklearn.linear_model import LogisticRegression

from six.moves.urllib.request import urlretrieve

from six.moves import cPickle as pickle

import cv2

import matplotlib.pyplot as plt

im_array = cv2.imread('../input/train/LAG/img_00091.jpg',0)

plt.imshow(im_array)
