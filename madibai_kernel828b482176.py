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

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from PIL import Image

import keras

from keras.layers import Dense, Activation, Flatten, Dropout

from keras.models import Sequential, Model

from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, Adam

import os

import tensorflow as tf

from keras import backend as K
data_train = pd.read_csv('train.csv')

data_train['flying'].hist()
train_names = data_train["Name"].values
train = []

for i in range(len(train_names)):

    try:

        file = Image.open('images1/' + train_names[i] + '.png').convert("RGB")

    except:

        file = Image.open('images1/' + train_names[i] + '.jpg').convert("RGB")

    

    values = np.array(file.getdata()).reshape(120, 120, 3)

    train.append(values)



train = np.array(train)