# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# reading train data

df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



# data shape

df.shape
#unique data types

print(df.dtypes.unique())
# exclude all the object type columns

n_df = df.select_dtypes(exclude='object')

print(n_df.shape)
# describe data

print(n_df.describe())
# data spliting

X = n_df.copy()

y = X.pop('SalePrice')

print(X.shape)

print(y.shape)
# count the missing values

print(X.isnull().sum())
# let's plot 'LotFrontage'

f, ax = plt.subplots(figsize=(20, 2))

sns.heatmap(X.corr().iloc[2:3,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax)

plt.show()
# plotting 

sns.kdeplot(X.LotFrontage, label='LotFrontage', shade=True)

# let's look details

X.LotFrontage.describe()
X['LotFrontage'].replace({np.nan:X.LotFrontage.mean()}, inplace=True)

print(X.LotFrontage.isnull().sum())
# correaltion with GrarageYrBlt

# print(X.GarageYrBlt.)

f, ax = plt.subplots(figsize=(20,2))

sns.heatmap(X.corr().iloc[25:26, :], annot=True, linewidths=0.8, fmt='.1f', ax=ax)

plt.show()
# let's describe

print(X.GarageYrBlt.describe())

#plotting

sns.kdeplot(X.GarageYrBlt, Label='GarageYrBlt', cbar=True, shade=True)
# fill missing value

X['GarageYrBlt'].replace({np.nan:X.GarageYrBlt.median()}, inplace=True)

X.GarageYrBlt.isnull().sum()
#let's see how impact on SalePrice

f, ax = plt.subplots(figsize=(20, 2))

sns.heatmap(X.corr().iloc[8:9,:], annot=True, linewidths=.8, fmt='.1f',ax=ax)

plt.show()
#plotting

sns.kdeplot(X.MasVnrArea, label='MasVnrArea', cbar=True, shade=True)

# describe

print(X.MasVnrArea.describe())
#replacing

X['MasVnrArea'].replace({np.nan:0}, inplace=True)

X.MasVnrArea.isnull().sum()
#let's check there is any missing value in X

X.isnull().sum()
# removing Id, since It won't be used for the prediction.

idx = X.pop('Id')

X.shape
f, ax = plt.subplots(figsize=(20, 2))

sns.heatmap(X.corr().iloc[35:36, :], annot=True, linewidths=.8, fmt='.1f', ax=ax)

plt.show()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#choosing k=30 best features

best_fit = SelectKBest(chi2, k=30).fit(X, y)

dfScores = pd.DataFrame(best_fit.scores_)

dfColumns = pd.DataFrame(X.columns)



features_score = pd.concat([dfColumns, dfScores], axis=1)

features_score.head()
#naming columns of features_score

features_score.columns = ['Class', 'Score']

features_score.head()
#print top 30 class according to score

print(features_score.nlargest(30, 'Score'))

features_score.shape
#final features

features = list(features_score.Class[:30])

print(len(features))
# update X data frame with features

X = X[features]

X.shape
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=4)

forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X, y)

predictions = forest_model.predict(val_X)

print(mean_absolute_error(val_y, predictions))
from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(X, y)

regr_preds = regr.predict(val_X)

print(mean_absolute_error(val_y, regr_preds))

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_data.head()
#igonre object data type

test_X = test_data.select_dtypes(exclude='object')

test_X = test_X.copy()

test_X.shape
# count missing value for each column

print(test_X.isnull().sum())
# for 'GarageYrBlt', 'MasVnrArea' and 'LotFrontage' doing the same for test data as i have done for train data 



test_X['GarageYrBlt'].replace({np.nan:test_X.GarageYrBlt.mean()}, inplace=True)

test_X['MasVnrArea'].replace({np.nan:0}, inplace=True)

test_X['LotFrontage'].replace({np.nan:test_X.LotFrontage.mean()}, inplace=True)
# count again missing values

print(test_X.isnull().sum())
## 1. BsmtFinSF1

print(test_X.BsmtFinSF1.describe())

sns.kdeplot(test_X.BsmtFinSF1, label='BsmtFinSF1', cbar=True, shade=True)

# since 50% value close to mean, I'm replacing with mean

test_X['BsmtFinSF1'].replace({np.nan:test_X.BsmtFinSF1.mean()}, inplace=True)

## 2. BsmtFinSF2

test_X.BsmtFinSF2.describe()
#since 75% value is zero, so, I'm replacing with zero

test_X['BsmtFinSF2'].replace({np.nan:0}, inplace=True)
## 3. BsmtUnfSF



print(test_X.BsmtUnfSF.describe())

sns.kdeplot(test_X.BsmtUnfSF, label='BsmtUnfSF', cbar=True, shade=True)
# replacing with 50% value (median value)

test_X['BsmtUnfSF'].replace({np.nan:test_X.BsmtUnfSF.median()}, inplace=True)
## 4. TotalBsmtSF

print(test_X.TotalBsmtSF.describe())

sns.kdeplot(test_X.TotalBsmtSF, label='TotalBsmtSF', cbar=True, shade=True)
#replacing with median value

test_X['TotalBsmtSF'].replace({np.nan:test_X.TotalBsmtSF.median()}, inplace=True)
## 5. BsmtFullBath

print(test_X.BsmtFullBath.describe())

sns.kdeplot(test_X.BsmtFullBath, label='BsmtFullBath', cbar=True, shade=True)

# replacing with mean/median, I'm using mean

test_X['BsmtFullBath'].replace({np.nan:test_X.BsmtFullBath.mean()}, inplace=True)
print(test_X.BsmtHalfBath.describe())

#since 75% values are zero, so I'm replacing with zero



test_X['BsmtHalfBath'].replace({np.nan:0}, inplace=True)
## 7. GarageCars

print(test_X.GarageCars.describe())

sns.kdeplot(test_X.GarageCars, label='GarageCars', cbar=True, shade=True)
#replacing with median

test_X['GarageCars'].replace({np.nan:test_X.GarageCars.median()}, inplace=True)
## 8. GarageArea

print(test_X.GarageArea.describe())

sns.kdeplot(test_X.GarageArea, label='GarageArea', cbar=True, shade=True)
#replacing with mean value

test_X['GarageArea'].replace({np.nan:test_X.GarageArea.mean()}, inplace=True)
# test data correlation

f, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(test_X.corr(), annot=True, linewidths=.8, fmt='.1f', ax=ax)

plt.show()
test_X = test_X[features]

test_X.isnull().sum()
test_X = preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_predictions = forest_model.predict(test_X)
# output for submission

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})

output.head()
# output to csv

output.to_csv('submission.csv', index=False)