# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt



%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train_V2.csv')
data_test = pd.read_csv('../input/test_V2.csv')
res_test = pd.read_csv('../input/sample_submission_V2.csv')
data_train[data_train['winPlacePerc']==1].head()
train = data_train[['boosts', 'kills', 'killPlace', 'heals']].head(5000)
result =  data_train['winPlacePerc'].head(5000)

min_train = train.head(5000)
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
scores_l = cross_val_score(linear_model.LinearRegression(), train, result, cv=8,  scoring='neg_median_absolute_error').mean()
scores_l


