# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import tensorflow as tf



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

import sklearn.model_selection as sk



import re



# Input data files are available in the "../input/" directory.

import os

print("Input files:")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# For neural nets with my GPU, RNN doesn't work without this in TF 2.0

from tensorflow.compat.v1 import ConfigProto

from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()

config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)



print()

print("TF Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")



if tf.test.gpu_device_name():

    print('GPU found')

else:

    print("No GPU found")
path_to_file = '/kaggle/input/wine-reviews/winemag-data-130k-v2.csv'



dfWine = pd.read_csv(path_to_file, index_col=0)
dfWine.info()

print()

print(dfWine.shape)

print(dfWine.columns)
dfWine.head(5)
dfWine.describe()
# # Removes the Twitter handles, that doesn't matter here

dfWine = dfWine.drop(['taster_twitter_handle'], axis=1)

dfWine.head(5)