# These are all the modules we'll be using later. Make sure you can import them

# before proceeding further.

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



# Config the matplotlib backend as plotting inline in IPython

%matplotlib inline