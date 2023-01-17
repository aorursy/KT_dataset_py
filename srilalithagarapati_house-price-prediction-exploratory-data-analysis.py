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



#Loading Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


#Load data from train data and test data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test =  pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()



# Display columns
train.columns
train['SalePrice'].describe()
sns.distplot(train['SalePrice'])
#Checking length of both train data and test data
print(len(train))
print(len(test))
#finding null values in train data and percentage of null values in each columns
Ntotal = train.isna().sum().sort_values(ascending = False)
percentage = (((train.isna().sum()/train.isna().count())*100).sort_values(ascending = False))
missing_data = pd.concat([Ntotal,percentage],axis = 1,keys =['Ntotal','percentage'])
missing_data.head(20)

#finding null values in test data
Ntotal =  test.isna().sum().sort_values(ascending = False)
percentage = (((test.isna().sum()/test.isna().count())*100).sort_values(ascending = False))
missing_data = pd.concat([Ntotal,percentage],axis = 1, keys = ['Ntotal','percentage'])
missing_data.head(10)
train = train.drop(columns = ['Id','PoolQC','MiscFeature','Alley','Fence'],axis = 1)
test = test.drop(columns = ['Id','PoolQC','MiscFeature','Alley','Fence'],axis = 1)
#Finding numerical and categorical columns in dataframe in both test and train data
train_num_cols = train.select_dtypes(exclude = 'object').columns
train_cat_cols = train.select_dtypes(include = 'object').columns


test_num_cols = test.select_dtypes(exclude = 'object').columns
test_cat_cols = test.select_dtypes(include = 'object').columns
#Display all numerical column names
train_num_cols
#Display all categorical column names
train_cat_cols
#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
#sales price correlation matrix
k = 10
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
crm = np.corrcoef(train[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
hm = sns.heatmap(crm, cbar=True, annot=True, square=True, fmt='.2f',cmap = 'viridis',linecolor = 'white', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols],size = 2.5)
plt.show()
var = 'GrLivArea'
data = pd.concat([train['SalePrice'],train[var]],axis =1)
data.plot.scatter(y = 'SalePrice',x = var,ylim =(0 ,800000))

var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'],train[var]],axis = 1)
data.plot.scatter(x = var,y = 'SalePrice',ylim = (0,800000) )
var = 'OverallQual'
data = pd.concat([train['SalePrice'],train[var]],axis = 1)
f,ax = plt.subplots(figsize = (8,10))
fig = sns.boxplot(x = var, y = 'SalePrice',data = data)
fig.axis(ymin = 0,ymax = 800000)
var = 'YearBuilt'
data = pd.concat([train['SalePrice'],train[var]],axis = 1)
f,ax = plt.subplots(figsize = (28,8))
fig = sns.boxplot(x=var, y = 'SalePrice',data = data)
fig.axis(ymin = 0,ymax = 800000)
train['Neighborhood'].value_counts()
all_houses =  train.shape[0]
print('Total number of houses:',all_houses)

#total number of houses in oldtown neighbourhood
houses_in_oldtown = train[train['Neighborhood'] == 'OldTown'].shape[0]
print('Total number of houses in oldtown neighbourhood:',houses_in_oldtown)
#probability of picking house in oldtown is
probability = (houses_in_oldtown/all_houses)*100
print('probability of picking house in oldtown is:{0:.2f}'.format(probability )+'%')