# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import csv
import seaborn as sns
import matplotlib.pyplot as plt
train_filename = "../input/home-data-for-ml-course/train.csv"

X_train_full = pd.read_csv(train_filename)

X_test_full = pd.read_csv("../input/home-data-for-ml-course/test.csv")

X_train_full.head()
X_test_full.head()
X_train_full.shape
y_train_full = X_train_full.SalePrice
X_train_full.columns
print(X_test_full.shape)
print(X_train_full.shape)
X_train_full.info();
X_train_full.isnull().sum()
X_train_full.LotFrontage
X_train_full['LotFrontage'].fillna(X_train_full['LotFrontage'].mean(),inplace = True)
X_train_full.isnull().sum()
neglig_col = ['PoolQC','Fence','MiscFeature','Alley','FireplaceQu']
for col in X_train_full.columns:
    if col in neglig_col:
        del X_train_full[col]
X_train_full.head()
X_train_num = X_train_full.select_dtypes(['int64','float64'])
X_train_categ = X_train_full.select_dtypes(['object'])
X_train_categ_cols = list(X_train_categ.columns)
X_train_num.head()
#X_train_categ_cols
X_train_num.isnull().any()
print("Pool Area Non Zero Values")
print(np.count_nonzero(X_train_num.PoolArea, axis=0))
print("Misc Val Non Zero Values")
print(np.count_nonzero(X_train_num.MiscVal,axis = 0))
print("3Ssn Porch Non Zero Values")
print(np.count_nonzero(X_train_num['3SsnPorch'], axis=0))
print("ScreenPorch Non Zero Values")
np.count_nonzero(X_train_num.ScreenPorch, axis=0)
#X_train.MiscVal.describe()
Almost_zero = ['3SsnPorch','ScreenPorch','PoolArea','MiscVal']
for col in X_train_num.columns:
    if col in Almost_zero:
        del X_train_num[col]
X_train_num.head()
X_train_categ.isnull().any()
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy = 'constant')
X_train_categ_imp = pd.DataFrame(imp.fit_transform(X_train_categ))
X_train_categ_imp.columns = X_train_categ.columns
X_train_categ_imp.isnull().any()
X_train_categ_imp.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
X_train_categ_imp.nunique()
#train_X_categ , valid_X_categ, train_y_categ , valid_y_categ = train_test_split(X_train_categ_imp,y_train_full, test_size = 0.2) 
cols_encode = []
for col in X_train_categ_imp:
    if X_train_categ_imp[col].nunique()>2:
        cols_encode.append(col)
label_X_train_categ = pd.get_dummies(X_train_categ_imp,columns=list(X_train_categ_imp.columns))
print(label_X_train_categ.shape)
label_X_train_categ.head()
label_X_train_categ.isnull()
X_train_num.info()
X_train_num.columns[X_train_num.isnull().any()]
X_train_num.MasVnrArea
np.count_nonzero(X_train_num.MasVnrArea, axis=0)
X_train_num.GarageYrBlt.describe()
num_fill = SimpleImputer(strategy = 'constant', fill_value = 0)
X_train_num_imp = pd.DataFrame(num_fill.fit_transform(X_train_num))
X_train_num_imp.columns = X_train_num.columns
X_train_num_imp.info()
X_train_num_imp.GarageYrBlt.describe()
X_train_num_imp.GarageYrBlt = X_train_num.GarageYrBlt
X_train_num_imp.GarageYrBlt.describe()
X_train_num_imp.GarageYrBlt.interpolate(method = 'nearest',inplace =True)
X_train_num_imp.GarageYrBlt.isnull().any()
print(X_train_num_imp.GarageYrBlt)
print("Number of unique entries in column "+str(X_train_num_imp.GarageYrBlt.nunique()))
print("Number of null entries in the column "+str(X_train_num_imp.GarageYrBlt.isna().sum()))
X_train_num_imp.info()
print("Number of null entries in the column "+str(X_train_num_imp.MasVnrArea.isna().sum()))
X_train_num_imp.MasVnrArea
np.count_nonzero(X_train_num_imp.MasVnrArea,axis =0)
label_X_train_categ.isnull().any()
label_X_train_categ
X_train_num_imp
tables = [X_train_num_imp,label_X_train_categ]
X_train = pd.concat(tables,axis = 1)
print(X_train.shape)
X_train
X_train.to_csv('Cleaned_train.csv')
X_test_full.isnull().sum()
neglig_col = ['PoolQC','Fence','MiscFeature','Alley','FireplaceQu']
for col in X_test_full.columns:
    if col in neglig_col:
        del X_test_full[col]
X_test_full.head()
X_test_num = X_test_full.select_dtypes(['int64','float64'])
X_test_categ = X_test_full.select_dtypes(['object'])
X_test_num.head()
X_test_categ.head()
X_test_num.info()
test_imp = SimpleImputer(strategy = 'mean')
X_test_num_imp = pd.DataFrame(test_imp.fit_transform(X_test_num))
X_test_num_imp.columns = X_test_num.columns
X_test_num_imp.info()
X_test_categ.info()
testcat_imp = SimpleImputer(strategy = 'constant')
X_test_categ_imp = pd.DataFrame(imp.fit_transform(X_test_categ))
X_test_categ_imp.columns = X_test_categ.columns
X_test_categ_imp.isnull().any()
X_test_categ_imp.head()
label_X_test_categ = pd.get_dummies(X_test_categ_imp,columns=list(X_test_categ_imp.columns))
label_X_test_categ.head()
test_tables = [X_test_num_imp,label_X_test_categ]
X_test = pd.concat(test_tables,axis = 1)
print(X_test.shape)
X_test
X_test.to_csv('Cleaned_test.csv')