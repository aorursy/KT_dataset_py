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
# import modules

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from scipy import stats
# load dataset

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



# shapes of the dataset

print('Shape of the train set = ', train.shape)

print('Shape of the test set = ', test.shape)



# description of the target

train['SalePrice'].describe()
# relation numerical variable - target (scatter)

plt.figure()

plt.scatter(train.GrLivArea, train.SalePrice, c='blue', marker='o', alpha=0.1)

plt.xlabel('Ground surface (square feeet)')

plt.ylabel('House price (dollars)')

plt.title('Quick outliers check')



# relation categorical variable - target (boxplot)

plt.figure()

sns.boxplot(x=train['OverallQual'], y=train['SalePrice'])



# remove outliers

train = train[train['GrLivArea'] < 4000]



# histogram of the target

plt.figure()

sns.distplot(train['SalePrice'])

# apply log transform since SalePrice is skewed, this will normally distribute the target variable, making the error of the predictions have the same variance for all ranges of the features values (homoscedasticity)

train.SalePrice = np.log(train.SalePrice)
# correlation matrix 

plt.subplots(figsize=(15,15))

sns.heatmap(train.corr())



# correlation matrix (top 10 features most correlated to SalePrice)

cols = train.corr().nlargest(10, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

plt.subplots(figsize=(15,15))

sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)



# scatter plot (top 10 features most correlated to SalePrice)q

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols])
# total number of samples missing for each feature

total = train.isnull().sum().sort_values(ascending = False)

# tercentage of samples missing for each feature

percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending = False) * 100

# stack total and percent series

missing = pd.concat([total, percent], axis = 1, keys = ['Total','Percent'])

# print summary of missing data

print(missing.head(30))

# remove features with missing data bigger than 0.5%

train.drop(missing[missing.Percent > 0.5].index, axis = 1, inplace = True)

test.drop(missing[missing.Percent > 0.5].index, axis = 1, inplace = True)

# print new dimensions

print('Train shape: ', train.shape, ' Test shape: ', test.shape)
# graph target versus GrLivArea

plt.scatter(train.GrLivArea, train.SalePrice, alpha = 0.1)
# convert categorical variables to binary

dataset = pd.get_dummies(pd.concat((train.iloc[:,:-1], test), axis = 0))

# extract datasets

trainset = pd.concat((dataset.iloc[:train.shape[0],:], train.SalePrice), axis = 1)

testset = dataset.iloc[train.shape[0]:,:]

# print shapes (must be equal minus one)

print('Train shape: ', trainset.shape, ' Test shape: ', testset.shape)

# fill nan in testset with most repeated values

testset.fillna(testset.mean(), inplace = True)
# get features and targets

Xtrain = trainset.iloc[:,:-1]

ytrain = trainset.iloc[:,-1]

Xtest = testset.iloc[:,:]

# print shapes (must be equal minus one)

print('Train features: ', np.shape(Xtrain), ' Test features: ', np.shape(Xtest), ' Train target: ', np.shape(ytrain))
# import regressor

from sklearn.linear_model import RidgeCV

# instance regressor with cross-validation

reg = RidgeCV(alphas = [1e-3, 1e-2, 1e-1, 1], scoring = 'neg_mean_squared_error', fit_intercept = True)

# fit model

reg.fit(Xtrain,ytrain)

# get predictions and take them back from log

predictions = np.exp(reg.predict(Xtest))

# format predictions to submission file

pd.DataFrame({'id': test.Id, 'SalePrice': predictions}).to_csv('my_submission.csv', index = False)