# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Packages

from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from sklearn.datasets import make_circles

from sklearn.model_selection import train_test_split



import tensorflow as tf

import keras

from keras.models import Sequential

from keras import *

import matplotlib.pyplot as plt





import seaborn as sns
# Load Aerial Cactus Images

sample_submission = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

train = pd.read_csv("../input/aerial-cactus-identification/train.csv")
train.head()
seed = 2020

np.random.seed(seed)
images = train[["id"]]

cacti_ind = train[["has_cactus"]]
train_images, test_images, train_cacti_ind, test_cacti_ind = train_test_split(images, cacti_ind, test_size=0.20, random_state=seed)
# Normalize pixel values to be between 0 and 1

train_images, sample_submission[["id"]] = train_images / 255.0, sample_submission[["id"]] / 255.0