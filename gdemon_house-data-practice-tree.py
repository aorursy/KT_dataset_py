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


from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

data = pd.read_csv("../input/train.csv", sep=",")

print(data.head(10))
def preprocess_features(data):
  selected_features = data[
    ["OverallQual",
     "GrLivArea",
     "TotalBsmtSF",
     "YearBuilt"]]

  return selected_features

data_pcrocessed = preprocess_features(data)
train_target = data.SalePrice
train_x, test_x, train_y, test_y = train_test_split(data_pcrocessed, train_target, test_size=0.2)
print(train_x.describe())
print(train_y.describe())
print(test_x.describe())
print(test_y.describe())
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_x, train_y)
test_predict = forest_model.predict(test_x)
print(mean_absolute_error(test_y, test_predict))