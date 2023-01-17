%load_ext autoreload
%autoreload 2

%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')

trainData.describe()
m = RandomForestRegressor(n_jobs=-1)
# The following code is supposed to fail due to string values in the input data
m.fit(trainData.drop('SalePrice', axis=1), trainData.SalePrice)
train_cats(trainData)
trainData.head()
trainData.dtypes
# get only column having numeric values
numericColumn = list(trainData.dtypes[trainData.dtypes != 'category'].index)

# creating a new dataframe not having categorial values 
numericDataFrame = trainData[numericColumn]

# fill NaN value with the mean of respective column
numericDataFrame.fillna(numericDataFrame.mean(), inplace = True)

numericDataFrame.head(5)
# https://www.analyticsvidhya.com/blog/2018/10/comprehensive-overview-machine-learning-part-1/
trainData, y, nas = proc_df(trainData, 'SalePrice')

m = RandomForestRegressor(n_jobs=-1)
m.fit(trainData, y)
m.score(trainData, y)
# Second Method - 
# https://www.analyticsvidhya.com/blog/2018/10/comprehensive-overview-machine-learning-part-1/
numericDataFrame, y, nas = proc_df(numericDataFrame, 'SalePrice')

m = RandomForestRegressor(n_jobs=-1)
m.fit(numericDataFrame, y)
m.score(numericDataFrame, y)