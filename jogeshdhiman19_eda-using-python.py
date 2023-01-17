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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
%matplotlib inline
data_train=pd.read_csv("../input/train.csv")
print(data_train.head())
data_train.columns
data_train.describe()
data_train.info()
# which colunm has the categorical features
categorical_features = data_train.select_dtypes(include = ["object"]).columns
categorical_features
# colunm which has the numerical features
numerical_features = data_train.select_dtypes(exclude = ["object"]).columns
numerical_features
# Differentiate numerical features and categorical features
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
data_train.columns[data_train.isnull().any()].tolist()
total = data_train.isnull().sum().sort_values(ascending=False)
percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)
#descriptive statistics summary
data_train['SalePrice'].describe()
#histogram and normal probability plot
sns.distplot(data_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(data_train['SalePrice'], plot=plt)
#histogram and normal probability plot
sns.distplot(data_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(data_train['GrLivArea'], plot=plt)
#histogram and normal probability plot
sns.distplot(data_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(data_train['TotalBsmtSF'], plot=plt)
#scatter plot grlivarea/saleprice

data_train.plot.scatter(x='GrLivArea', y='SalePrice');
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data_train.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice
var = 'OverallQual'
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data_train)
fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data_train)
fig.axis(ymin=0, ymax=800000);
#correlation matrix
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data_train[cols], size = 2.5)
plt.show();
