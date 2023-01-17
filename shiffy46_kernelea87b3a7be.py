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
import xgboost as xgb

from sklearn.model_selection import cross_validate

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import xgboost

import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_rows',1000)
xg_reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.08, max_delta_step=0,

             max_depth=3, min_child_weight=1, missing=None, n_estimators=100,

             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

for i in range(3,21):

    train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

    noNullList = list([train.isna().sum() != 0])[0].index

    train = train[noNullList]

    goodCols = list(train.corr()['SalePrice'].nlargest(i).index)

    train = train[goodCols]

    X = train.iloc[:,1:]

    y = train.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

    scores = cross_validate(xg_reg, X, y, cv=3,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)

    #xg_reg.fit(X_train,y_train)

    #score = xg_reg.score(X_test,y_test)

    print("for nlargest: "+str(i)+' score is: '+str(scores['test_r2']))
    train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

    noNullList = list([train.isna().sum() != 0])[0].index

    train = train[noNullList]

    goodCols = list(train.corr()['SalePrice'].nlargest(21).index)

    train = train[goodCols]

    scaler = StandardScaler()

    X = train.iloc[:,1:]

    y = train.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

    X_train = scaler.fit_transform(X_train)

    xg_reg.fit(X_train,y_train)

    scores = cross_validate(xg_reg, X, y, cv=3,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)['test_r2']
scores
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
noNullList = list([train.isna().sum() != 0])[0].index

train = train[noNullList]

goodCols = list(train.corr()['SalePrice'].nlargest(20).index)

train = train[goodCols]
train
X = train.iloc[:,1:]

y = train.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

xg_reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.08, max_delta_step=0,

             max_depth=3, min_child_weight=1, missing=None, n_estimators=100,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

xg_reg.fit(X_train,y_train)
xg_reg.score(X_test,y_test)
testingDF = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

testingDF2 = testingDF[goodCols[1:]]

testingDF2 = scaler.fit_transform(testingDF2)
preds = xg_reg.predict(testingDF2)

predDF = pd.DataFrame()

predDF['Id'] = testingDF['Id']

predDF['SalePrice'] = pd.DataFrame(preds)
predDF.to_csv('submission.csv',index=False)
predDF
from sklearn.model_selection import cross_validate

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score



from math import sqrt
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

id = train.Id

noNullList = list([train.isna().sum() != 0])[0].index

train = train[noNullList]

train2 = train.select_dtypes(exclude=['object'])

goodCols = list(train2.corr()['SalePrice'].nlargest(10).index)

# goodCols = list(train.corr()['SalePrice'].nlargest(20).index)

print(train2.dtypes)

# print(train.select_dtypes(exclude = ['object']))

# print(train[goodCols])

train2 = train2[goodCols]

X = train2.iloc[:,1:]

y = train2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y)
goodCols[1:]
randForestModel = RandomForestRegressor()



randForestModel.fit(X_train, y_train)
prediction = randForestModel.predict(X_test)

# print(prediction

# Ours

print("RandomForestRegressor RMSE:",sqrt(mean_squared_error(y_test, prediction)))
train
my_submission = pd.DataFrame()

my_submission['Id'] = train['Id']

my_submission['SalePrice'] = prediction

my_submission.to_csv('submission.csv', index=False)
noNullList
testingData2.isna().sum()
testingData = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

COLS = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath','TotRmsAbvGrd','YearBuilt']

testingData = testingData[noNullList[:-1]]

testingData2 = testingData[COLS]

imp_mean.fit(testingData2)

testingData2 = imp_mean.transform(testingData2)

predictions = randForestModel.predict(testingData2)
my_submission = pd.DataFrame()

my_submission['Id'] = testingData['Id']

my_submission['SalePrice'] = predictions

#my_submission.to_csv('submission.csv', index=False)
len(testingData2)