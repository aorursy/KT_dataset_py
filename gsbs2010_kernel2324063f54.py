# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras

from keras.layers import Dense, Conv2D, BatchNormalization, Activation

from keras.layers import AveragePooling2D, Input, Flatten

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.callbacks import ReduceLROnPlateau

from keras.regularizers import l2

from keras import backend as K

from keras.models import Model

from keras.utils import to_categorical

import numpy as np

import os

from sklearn.model_selection import train_test_split

from zipfile import ZipFile

import matplotlib.pyplot as plt



import PIL.Image

from IPython.display import display
def load_image(filename):

    image = PIL.Image.open(filename) # open colour image

    image = np.array(image)

    

    return np.int32(image)