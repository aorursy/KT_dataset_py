# Data file

import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Import 

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns 



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.linear_model import RANSACRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import LassoCV

from sklearn.linear_model import ElasticNet

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

from sklearn.svm import LinearSVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost.sklearn import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import VotingRegressor
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', header=0)

train.head(10)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test.head(10)
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission.head(10)
train_mid = train.copy()

train_mid['train_or_test'] = 'train'



test_mid = test.copy()

test_mid['train_or_test'] = 'test'



test_mid['SalePrice'] = 9



alldata = pd.concat([train_mid, test_mid], sort=False, axis=0).reset_index(drop=True)



print('The size of the train data:' + str(train.shape))

print('The size of the test data:' + str(test.shape))

print('The size of the submission data:' + str(submission.shape))

print('The size of the alldata data:' + str(alldata.shape))
# Check dtype and missing value

print('=====Train=====')

train.info()

print('\n=====Test=====')

test.info()
# The basic statistics of the dataset are also checked

train.describe()

test.describe()

alldata.describe()
# Check for duplicates ID

idsUnique = len(set(alldata['Id']))

idsTotal = alldata.shape[0]

idsDupli = idsTotal - idsUnique

print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")
# Drop PassengerId column. del alldata['Id']

alldata.drop("Id", axis = 1, inplace = True)
# Missing data

def Missing_table(df):

    # null_val = df.isnull().sum()

    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

    list_type = df[na_col_list].dtypes.sort_values(ascending=False)

    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'Missing data', 1:'%', 2:'type'})

    return missing_table_len.sort_values(by=['Missing data'], ascending=False)



Missing_table(train)
# Missing data

def Missing_table(df):

    # null_val = df.isnull().sum()

    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

    list_type = df[na_col_list].dtypes.sort_values(ascending=False)

    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'Missing data', 1:'%', 2:'type'})

    return missing_table_len.sort_values(by=['Missing data'], ascending=False)



Missing_table(test)
# Missing data

def Missing_table(df):

    # null_val = df.isnull().sum()

    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

    list_type = df[na_col_list].dtypes.sort_values(ascending=False)

    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'Missing data', 1:'%', 2:'type'})

    return missing_table_len.sort_values(by=['Missing data'], ascending=False)



Missing_table(alldata)
train['WhatIsData'] = 'Train'

test['WhatIsData'] = 'Test'

test['SalePrice'] = 9999999999

alldata = pd.concat([train,test],axis=0).reset_index(drop=True)



# Making train data feature as list

cat_cols = alldata.dtypes[train.dtypes=='object'].index.tolist()

num_cols = alldata.dtypes[train.dtypes!='object'].index.tolist()



other_cols = ['Id','WhatIsData']

cat_cols.remove('WhatIsData')

num_cols.remove('Id')



cat = pd.get_dummies(alldata[cat_cols])



# Marge data

all_data = pd.concat([alldata[other_cols],alldata[num_cols].fillna(0),cat],axis=1)

all_data.describe()

# Matrix

import matplotlib.pyplot as plt

corrmat = train.corr()

f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(corrmat, vmax=.8, square=True)
# Extract the items that are highly relevant to SalsPrice

import matplotlib.pyplot as plt

corrmat = train.corr()

# In descending order of relevance

k = 21 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

plt.figure(figsize=(15, 15))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

cols
# Display of scatterplots of highly correlated OverallQual and GrLivArea

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(train[cols], size = 2.5)

plt.show()
# Looking at the SalesPrice and GrLivArea scatterplots in the top right corner, there are two data that are significantly out of trend. Remove this data as improper training data or outlier data. Show the top two highest numbered data

train.sort_values(by = 'GrLivArea', ascending = False)[:2]
# Delete outstanding 2 datas

train = train.drop(index = train[train['Id'] == 1299].index)

train = train.drop(index = train[train['Id'] == 524].index)



# Scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(train[cols], size = 2.5)

plt.show()