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

from keras.applications.vgg19 import VGG19

from keras.models import Model

from keras.layers import Dense, Dropout, Flatten

import tensorflow as tf

import os

from tqdm import tqdm

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from fastai.metrics import error_rate
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
PATH ="../input"
!ls {PATH}
df_train = pd.read_csv('../input/labels.csv')

df_test = pd.read_csv('../input/sample_submission.csv')
df_train.head(10)
df_train.shape
df_train.breed.value_counts().head(5)
df_train.columns
df_train.describe()
df_test.shape
df_test.columns
df_test.id.value_counts().head(3)
df_test.describe()
targets_series = pd.Series(df_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)
im_size = 90
x_train = []

y_train = []

x_test = []
i = 0 

for f, breed in tqdm(df_train.values):

    img = cv2.imread('../input/train/{}.jpg'.format(f))

    label = one_hot_labels[i]

    x_train.append(cv2.resize(img, (im_size, im_size)))

    y_train.append(label)

    i += 1
for f in tqdm(df_test['id'].values):

    img = cv2.imread('../input/test/{}.jpg'.format(f))

    x_test.append(cv2.resize(img, (im_size, im_size)))
y_train_raw = np.array(y_train, np.uint8)

x_train_raw = np.array(x_train, np.float32) / 255.

x_test  = np.array(x_test, np.float32) / 255.
print(x_train_raw.shape)

print(y_train_raw.shape)

print(x_test.shape)
num_class = y_train_raw.shape[1]
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)