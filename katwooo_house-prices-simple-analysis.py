import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import mlxtend



pd.set_option('display.max_columns', 100, 'display.max_rows', 500)
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print(len(train))

print(len(test))
train = train.sort_index(axis='columns')

test = test.sort_index(axis='columns')
train.head()
test.tail()
all_data = pd.concat([train, test], ignore_index=True)

all_data.isnull().sum()[all_data.isnull().sum() > 0]
corr = all_data.corr()

plt.figure(figsize=(40,40))

sns.heatmap(data=corr, annot=True)
na_col = all_data.isnull().sum()[all_data.isnull().sum() > 0].index.tolist()

all_data[na_col].dtypes.sort_index()
numeric_data = train.select_dtypes(include=[np.number])

corr = numeric_data.corr()

corr['SalePrice'].sort_values(ascending=False)*100
pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

pivot
for col in ['Alley', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Fence',\

               'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'MasVnrType', 'MiscFeature', 'PoolQC']:

    all_data[col] = all_data[col].fillna('NA')

for col in ['Electrical', 'Exterior1st', 'Exterior2nd', 'FireplaceQu', 'Functional', 'KitchenQual',\

               'MSZoning', 'SaleType', 'Utilities']:

    mode = all_data[col].mode()[0]

    all_data[col] = all_data[col].fillna(mode)



print(all_data.isnull().sum()[all_data.isnull().sum() > 0])
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda d:d.fillna(d.median()))

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0) #null = MasVnrArea無し

all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)

all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)

all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)

all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)

all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)

all_data['GarageArea'] = all_data['GarageArea'].fillna(0)

all_data['GarageCars'] = all_data['GarageCars'].fillna(0)

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data[all_data['GarageYrBlt'].isnull()]['YearBuilt']) 

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
all_data.isnull().sum()[all_data.isnull().sum() > 0]
train.head()
y = train['SalePrice']
y.hist(bins=40)
y_log = np.log1p(y)

y_log.hist(bins=40)
all_data.dtypes.sort_values()
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['YrSold'] = all_data['YrSold'].apply(str)

all_data['MoSold'] = all_data['MoSold'].apply(str)
all_data.dtypes.sort_values()
train_ID = train['Id']

test_ID = test['Id']

print(len(train_ID))

print(len(test_ID))
all_data = pd.get_dummies(all_data, drop_first=True)

all_data
all_data.pop('SalePrice')

X = all_data.iloc[:1460]

XX = all_data.iloc[1460:]

y = y_log
X
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    random_state=0)
# Prints R2 and RMSE scores

def get_score(prediction, lables):    

    print('R2: {}'.format(r2_score(prediction, lables)))

    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))





# Shows scores for train and validation sets    

def train_test(estimator, x_trn, x_tst, y_trn, y_tst):

    prediction_train = estimator.predict(x_trn)

    # Printing estimator

    print(estimator)

    # Printing train scores

    get_score(prediction_train, y_trn)

    prediction_test = estimator.predict(x_tst)

    # Printing test scores

    print("Test")

    get_score(prediction_test, y_tst)
from sklearn.ensemble import GradientBoostingRegressor

GBest = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=0).fit(X_train, y_train)

train_test(GBest, X_train, X_test, y_train, y_test)
## Getting our SalePrice estimation

y_GB = (np.exp(GBest.predict(XX)))
# y_ens

y_GB
submission = pd.DataFrame()

submission['Id'] = test['Id']

submission['SalePrice'] = y_GB
submission.head()
submission.to_csv('submission_12493.csv', index =False)