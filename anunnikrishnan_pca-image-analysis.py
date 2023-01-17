# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
mnist.data.shape
from sklearn.model_selection import train_test_split

# test_size: what proportion of original data is used for test set

train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=0.3, random_state=0)
train_img.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training set only.

scaler.fit(train_img)

# Apply transform to both the training set and the test set.

train_img = scaler.transform(train_img)

test_img = scaler.transform(test_img)
from sklearn.decomposition import PCA

# Make an instance of the Model

pca = PCA(n_components= 4)
pca.fit(train_img)
train_img = pca.transform(train_img)

test_img = pca.transform(test_img)
pca.explained_variance_ratio_
train_img.shape
train_lbl.shape
train_lbl
from sklearn.ensemble import RandomForestClassifier

# all parameters not specified are set to their defaults

rf = RandomForestClassifier()

rf.fit(train_img, train_lbl)

# Predict for One Observation (image)

rf.predict(test_img[0].reshape(1,-1))

# Predict for One Observation (image)

rf.predict(test_img[0:10])
import matplotlib.pyplot as plt

plt.imshow(mnist.data[6].reshape(28,28),

              cmap = plt.cm.gray, interpolation='nearest',

              clim=(0, 255));
import matplotlib.pyplot as plt

plt.imshow(mnist.data[5].reshape(28,28),

              cmap = plt.cm.gray, interpolation='nearest',

              clim=(0, 255));