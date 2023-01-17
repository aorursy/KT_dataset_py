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
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/data.csv')
data.head()
data.describe()
data.info()
y=data['target']
X=data.drop(['target','artist','song_title'],axis=1)

train_x,test_x,train_y,test_y=train_test_split(X.as_matrix(),y.as_matrix(),test_size=0.15)
print("Training size\t:\t",len(train_x),"\nTest Size\t:\t",len(test_x))
train_x.shape
from xgboost import XGBRegressor
my_model = XGBRegressor()
my_model.fit(train_x,train_y)
predictions = my_model.predict(test_x)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(test_y,predictions))

