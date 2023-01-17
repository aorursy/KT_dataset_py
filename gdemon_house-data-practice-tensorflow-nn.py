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
FEATURES = ["OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt"]
LABEL = "SalePrice"
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

data_pcrocessed = preprocess_features(data)
train_target = data.SalePrice
train_x, test_x, train_y, test_y = train_test_split(data_pcrocessed, train_target, test_size=0.2)

train_y = pd.DataFrame(train_y, columns = [LABEL])
training_set = pd.DataFrame(train_x, columns = FEATURES).merge(train_y, left_index = True, right_index = True)

#print(train_x.describe())
#print(train_y.describe())
#print(test_x.describe())
#print(test_y.describe())
# Same thing but for the test set
test_y = pd.DataFrame(test_y, columns = [LABEL])
testing_set = pd.DataFrame(test_x, columns = FEATURES).merge(test_y, left_index = True, right_index = True)
testing_set.head()
# Model
tf.logging.set_verbosity(tf.logging.ERROR)
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])
def input_fn(data_set, pred = False):
    
    if pred == False:
        
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        
        return feature_cols, labels

    if pred == True:
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        
        return feature_cols

# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)

# Evaluation on the test set created by train_test_split
eval = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)
# Display the score on the testing set
# 0.002X in average
loss_score1 = eval["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score1))
import itertools

# Predictions
y = regressor.predict(input_fn=lambda: input_fn(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))