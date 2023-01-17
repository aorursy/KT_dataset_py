import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.drop(columns=['Id'], inplace = True)
train.head()
train.isnull().sum().sort_values(ascending=False)
train.drop(columns = ['PoolQC', 'MiscFeature','Alley', 'Fence'], inplace = True)
list_has_null_values= ['FireplaceQu','LotFrontage','GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType','Electrical']
for col in list_has_null_values:

    if (train[col].dtype == np.object):

        train[col].fillna(0, inplace = True)

    else:

        train[col].fillna(train[col].median(), inplace = True)
object_columns = list(train.select_dtypes(include = ['object']).columns)
train_encoded = pd.get_dummies(train.iloc[:, :-1], columns=object_columns)
X =train_encoded.iloc[:,:-1]

y = train.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size = 0.3, random_state=1)
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=1, max_depth=8, min_samples_leaf=0.1)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_valid)
from sklearn.metrics import mean_squared_error as MSE
mse_dt = MSE(y_valid,y_pred_dt)

print(mse_dt)
rmse_dt = mse_dt**(1/2)

print(rmse_dt)
from xgboost.sklearn import XGBRegressor

xgb = XGBRegressor()

xgb.fit(X_train, y_train)

y_preds_xgb = xgb.predict(X_valid)
from sklearn.metrics import r2_score
r2_score(y_valid, y_preds_xgb)
test.head()
test.drop(columns = ['PoolQC', 'MiscFeature','Alley', 'Fence'], inplace = True)
test_has_null_columns = test.isnull().sum().sort_values(ascending=False)[:30]
test_has_null_columns
test_has_null_columns = ['FireplaceQu',

'LotFrontage',

'GarageCond',

'GarageQual',

'GarageYrBlt',

'GarageFinish',

'GarageType',

'BsmtCond',

'BsmtQual',

'BsmtExposure',

'BsmtFinType1',

'BsmtFinType2',

'MasVnrType',

'MasVnrArea',

'MSZoning',

'BsmtHalfBath',

'Utilities',

'Functional',

'BsmtFullBath',

'BsmtUnfSF',

'SaleType',

'BsmtFinSF2',

'BsmtFinSF1',

'Exterior2nd',

'Exterior1st',

'TotalBsmtSF',

'GarageCars',

'KitchenQual',

'GarageArea']
for col in test_has_null_columns:

    if (test[col].dtype == np.object):

        test[col].fillna(0, inplace = True)

    else:

        test[col].fillna(test[col].median(), inplace = True)
test_object_columns = list(test.select_dtypes(include=['object']).columns)

test_encoded = pd.get_dummies(test,columns=test_object_columns)

test_encoded
for col in test_encoded.columns:

    if (col not in X.columns):

        test_encoded.drop(columns=[col], inplace=True)

X.shape, test_encoded.shape
for col in X.columns:

    if (col not in test_encoded.columns):

        test_encoded[col]=0

X.shape, test_encoded.shape
y_preds_dt_res = dt.predict(test_encoded)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_preds_dt_res})

my_submission.to_csv("y_pred_dt_res.csv", index=False)