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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.columns
print('length of train and test columns are: ' + str(len(train.columns)) + ' and ' + str(len(test.columns)) + ' respectively.')
train.head()
missing_in_test = [col for col in train.columns
    if col not in test.columns ]
print('missing column in test data is ' + missing_in_test[0])
print("Ofc this is quite obnious fact i missed out. let's get going forward")
train.describe()
print('number of numerical columns are: ' + str(len(train.describe().columns)))
print('Shapes of Train and Test Datasets are - Train: ' + str(train.shape) + ', Test: ' + str(test.shape))
print("Unique no of ID's: " + str(len(set(train.Id))))
print("Duplicate ID's: " + str(len(set(train.Id)) - train.shape[0]))
# lets drop Id column since it doesn't have any duplicate id's
train.drop(['Id'], axis=1, inplace=True)
# correlation matrix ( check only for numerical columns as far as i understand )
corr_matrix = train.corr()
top_corr_features = corr_matrix.index[abs(corr_matrix['SalePrice'] > 0.5)]
top_corr_features
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10, 10))
g = sns.heatmap(train[top_corr_features].corr(), annot=True, cmap='RdYlGn')
#OverallQual, GrLivArea, GarageCars, GarageArea have high correlation with SalePrice
sns.barplot(train.OverallQual, train.SalePrice)
# We can see the increase in Price with increase in Overall Quality of the houses
sns.set()
effective_numeric_cols = ['SalePrice', 'OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'YearBuilt']
sns.pairplot(train[effective_numeric_cols], size=2.5)
plt.show()
sns.distplot(train.SalePrice)
print("Skewness : %f \nKurtosis : %f " %(train['SalePrice'].skew(), train['SalePrice'].kurt()))
# train.columns
train_missing_total = train.isnull().sum().sort_values(ascending=False)
train_missing_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([train_missing_total, train_missing_percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# remove features which have more than 15% missing values and doesn't contribute much to our model in predicting sales prices
# fill the missing values with any central tendancy you seem fit or using other missing value treatments for the rest of features which matter and have less than 15% missing values
train = train.drop((missing_data[missing_data['Total']>1]).index, 1)

train = train.drop(train.loc[train['Electrical'].isnull()].index)
# removed some columns above
train.shape
# max no of null values in columns is 0, so no null values in dataframe
# cross Checking purpose
train.isnull().sum().max()
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
# missing_data[missing_data['Total']>1].index
# returns Index(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage',
#        'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageYrBlt',
#        'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1',
#        'MasVnrArea', 'MasVnrType'],
#       dtype='object')  --- data where missing data is more than 1.

# and now that there are no missing values, let's check for outliers. 
# Outlier Treatment
from sklearn.preprocessing import StandardScaler
train['SalePrice'][:, np.newaxis]
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:, np.newaxis])
saleprice_scaled
low_range_values = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range_values = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('lower side outliers of salesprice: \n' + str(low_range_values))
print('higher side outliers of salesprice: \n' + str(high_range_values))
# bivariate analysis for GrlivArea and Saleprice - Scatter Plot
Gr_Sp_data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
Gr_Sp_data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
# remove the two outliers shown above in scatter plot
train.drop(train.sort_values(by = 'GrLivArea', ascending = False)[:2].index, inplace=True)
# bivariate analysis for 'TotalBsmtSF'
Tbsmt_Sp_data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
Tbsmt_Sp_data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000))
from scipy import stats
from scipy.stats import norm
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
# positive skewness observed for sale price distribution
# lets use log on the values to eliminate this skewness
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
# now plots follow the normal distrubution and skewness
#histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
train['GrLivArea'] = np.log(train['GrLivArea'])
#histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)

# similarly for GrLivArea
#histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
# Something that, in general, presents skewness.
# A significant number of observations with value zero (houses without basement).
# A big problem because the value zero doesn't allow us to do log transformations.

# To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement (binary variable). 
# Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. 
# This way we can transform data, without losing the effect of having or not basement.

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0

# should have written 0 inside the Series rather than assigning 0 in the second line
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

#histogram and normal probability plot for non zero values
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#convert categorical variable into dummy
train = pd.get_dummies(train)
train.columns
# after converting categorical features into dummies
y = train.SalePrice
X = train.drop(['SalePrice'], axis=1)

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
# melb_preds = forest_model.predict(test)
# print(mean_absolute_error(val_y, melb_preds))

# get predicted prices on validation data
val_predictions = forest_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
