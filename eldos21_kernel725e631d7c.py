# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as st

import matplotlib.pyplot as plt # plot visualization

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.columns
train.shape
missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
train.drop('PoolQC', axis=1, inplace=True)

train.drop('MiscFeature', axis=1, inplace=True)

train.drop('Fence', axis=1, inplace=True)

train.drop('Alley', axis=1, inplace=True)

train.drop('FireplaceQu', axis=1, inplace=True)
test.drop('PoolQC', axis=1, inplace=True)

test.drop('MiscFeature', axis=1, inplace=True)

test.drop('Fence', axis=1, inplace=True)

test.drop('Alley', axis=1, inplace=True)

test.drop('FireplaceQu', axis=1, inplace=True)
train['SalePrice'].describe()
correlation = train.select_dtypes(include=[np.number]).corr()

print(correlation['SalePrice'].sort_values(ascending = False))
train.drop('KitchenAbvGr', axis=1, inplace=True);train.drop('EnclosedPorch', axis=1, inplace=True);train.drop('MSSubClass', axis=1, inplace=True)

train.drop('OverallCond', axis=1, inplace=True);train.drop('YrSold', axis=1, inplace=True);train.drop('LowQualFinSF', axis=1, inplace=True)

train.drop('Id', axis=1, inplace=True);train.drop('MiscVal', axis=1, inplace=True);train.drop('BsmtHalfBath', axis=1, inplace=True)

train.drop('BsmtFinSF2', axis=1, inplace=True)
test.drop(columns=['KitchenAbvGr', 'EnclosedPorch', 'MSSubClass', 'OverallCond', 'YrSold', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2'], axis=1, inplace=True)
y = train['SalePrice']

plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)
sns.set_style('whitegrid')

st.probplot(train['SalePrice'], plot=plt)

plt.xlabel('SalePrice', fontsize=15)

plt.title('Probability plot', fontsize=15)

plt.show()
train['SalePrice'] = train['SalePrice'].apply(np.log)
yn = train['SalePrice']

plt.title('Normal(trn.)')

sns.distplot(yn, kde=False, fit=st.norm)
sns.set_style('whitegrid')

st.probplot(train['SalePrice'], plot=plt)

plt.xlabel('SalePrice', fontsize=15)

plt.title('Probability plot', fontsize=15)

plt.show()
categorical_features = train.select_dtypes(include = ["object"]).columns

categorical_features
from sklearn.preprocessing import LabelEncoder
for c in categorical_features:

    lbl = LabelEncoder() 

    lbl.fit(list(train[c].values)) 

    train[c] = lbl.transform(list(train[c].values))
test_cat = test.select_dtypes(include = ["object"]).columns

for c in test_cat:

    lbl = LabelEncoder() 

    lbl.fit(list(test[c].values)) 

    test[c] = lbl.transform(list(test[c].values))
missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
train['MasVnrArea'].fillna((train['MasVnrArea'].mean()), inplace=True)

train['GarageYrBlt'].fillna((train['GarageYrBlt'].mean()), inplace=True)

train['LotFrontage'].fillna((train['LotFrontage'].mean()), inplace=True)
test = test.fillna(test.mean())
train.head()
y = train.SalePrice.values

X = train.drop(['SalePrice'], axis=1).values
xtest = test.drop(['Id'], axis=1).values
X.shape, y.shape, xtest.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)
y_preds = reg.predict(xtest)
sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub['SalePrice'] = y_preds



sub.head()
sub.to_csv('output.csv', index=False)