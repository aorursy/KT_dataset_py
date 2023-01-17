# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


House = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

House_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
House.info()
House_test.info()
import math

House["SalePrice_log"] = np.log(House['SalePrice'])
pd.set_option('display.max_columns', None)

House.head(20)
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white")



corr = House.corr()



mask = np.triu(np.ones_like(corr, dtype=np.bool))



f, ax = plt.subplots(figsize=(35, 35))





cmap = sns.diverging_palette(220, 10, center = 'light',as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=0.5)
House = House.drop(columns =['Id', 'YrSold','MSSubClass','OverallCond','LowQualFinSF', 'BsmtHalfBath', 'Alley','MiscVal','MiscFeature','Fence'])
House = House.fillna("0")

House_test = House_test.fillna("0")
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
House = House.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
OneHotEncoder(handle_unknown='ignore').fit_transform(House)
House_test = House_test.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')

OneHotEncoder(handle_unknown='ignore').fit_transform(House_test)

#House_test.head(20)

House_test = House_test.drop(columns =['Id', 'YrSold','MSSubClass','OverallCond','LowQualFinSF', 'BsmtHalfBath', 'Alley','MiscVal','MiscFeature','Fence'])
from sklearn import linear_model

from sklearn.linear_model import LinearRegression

import random

regr = LinearRegression()

df1=House.drop(columns = ['SalePrice','SalePrice_log'])

Y1 = House["SalePrice_log"]

regr.fit(df1, Y1)

X2 = House_test

biases = [random.uniform(0,1) for j in range(len (X2))]

target = regr.predict(X2)

House_test["SalePrice"] = target

print(target)
from sklearn.metrics import mean_squared_error

from sklearn import metrics

print('Mean Squared Error is: ', metrics.mean_squared_error(House_test["SalePrice"], Y1[1:]))
from scipy import stats

residuals = Y1[1:] - target

res = stats.probplot(residuals,plot = plt)