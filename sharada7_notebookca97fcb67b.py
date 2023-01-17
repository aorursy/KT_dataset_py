# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline
from __future__ import print_function

import matplotlib.pyplot as plt

import numpy as np

import scipy.misc

import os

import sys

import tarfile

import random

from IPython.display import display, Image

from scipy import ndimage

from IPython.display import SVG

from keras.models import model_from_json

from keras.utils import np_utils

from keras.utils.visualize_util import model_to_dot

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.preprocessing import image