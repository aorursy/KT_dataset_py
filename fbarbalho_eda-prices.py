# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.shape
# looking at a few rows of data
train.sample(10)
# get the percent of missing values for each column 
perc_na =  (train.isnull().sum()/len(train))*100
ratio_na = perc_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :ratio_na})
missing_data.head(20)
collumns_to_drop = ['Id','PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage']
train.drop(collumns_to_drop, axis=1, inplace=True)

train.shape
#list all the numeric columns
numeric_variables = list(train.select_dtypes(include=['int64', 'float64']).columns.values)
# list all the non-numerical columns
categorial_variables = list(train.select_dtypes(exclude=['int64', 'float64', 'bool']).columns.values)


# Use one-hot encoding for categorical variables

new_data = pd.get_dummies(train, columns=categorial_variables)

new_data.sample(10)
# Imputation for numeric variables use median
from sklearn.preprocessing import Imputer
my_imputer = Imputer(strategy="median")
data_imputed = pd.DataFrame(my_imputer.fit_transform(new_data))
# add columns to new_data
data_imputed.columns = new_data.columns

# get the percent of missing values for each column 
perc_na =  (data_imputed.isnull().sum()/len(new_data))*100
ratio_na = perc_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :ratio_na})
missing_data.sum() 
#histogram
data_imputed.plot.scatter(x='GrLivArea', y='SalePrice')
# Choose target and predictors
y = data_imputed['SalePrice']
# Except SalePrice
X = data_imputed.drop(columns = 'SalePrice')
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
house_preds = forest_model.predict(val_X)
print('Mean absolute error: {}'.format(mean_absolute_error(val_y, house_preds)))
from sklearn import svm
clf = svm.SVC()
clf.fit(train_X, train_y)  

svm_house_preds = clf.predict(val_X)
print('Mean absolute error: {}'.format(mean_absolute_error(val_y, svm_house_preds)))


