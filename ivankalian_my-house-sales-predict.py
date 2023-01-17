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
import pandas as pd

import numpy as np

import warnings

from sklearn import preprocessing

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train_x = ['Id','SalePrice', 'OverallQual', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars']

true_col_x = []

false_col_y = []

for i in df_train.columns:

    if df_train[i].count() == 1460 and i not in train_x:

        true_col_x.append(i)

    elif df_train[i].count() < 1460 and i not in train_x:

        false_col_y.append(i)
for i in true_col_x:

    le = preprocessing.LabelEncoder()

    le.fit(df_train[i])

    df_train[i] = le.transform(df_train[i])
# проверка незаполненных столбцов

total = df_train.isnull().sum()

procent = df_train.isnull().sum() / df_train.notnull().sum()

missing_data = pd.concat([total, procent], axis=1, keys=['Total', 'Percent'])

# pd.concat([missing_data[missing_data['Percent'] > 0], missing_data[missing_data['Percent'] != 0.000685]], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Percent'] > 0].count()
df_train = df_train.drop(false_col_y, axis = 1)

target = np.log(df_train['SalePrice']) #log распределение

df_train = df_train.drop(['Id', 'SalePrice'], axis=1)
df_train['GarageCars'] = df_train['GarageCars'].astype('float64')

df_train['TotalBsmtSF'] = df_train['TotalBsmtSF'].astype('float64')
df_train.shape
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_x = ['Id', 'OverallQual', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars']

true_col_test_x = []

false_col_test_y = []

for i in df_test.columns:

    if df_test[i].count() == 1459 and i not in test_x:

        true_col_test_x.append(i)

    elif df_test[i].count() < 1459 and i not in test_x:

        false_col_test_y.append(i)
df_test['SaleType'][df_test['SaleType'].isnull()] = 'WD'

df_test['MSZoning'][df_test['MSZoning'].isnull()] = 'RL'

df_test['Utilities'][df_test['Utilities'].isnull()] = 'AllPub'

df_test['Exterior1st'][df_test['Exterior1st'].isnull()] = 'VinylSd'

df_test['Exterior2nd'][df_test['Exterior2nd'].isnull()] = 'VinylSd'

df_test['BsmtFinSF1'][df_test['BsmtFinSF1'].isnull()] = 0.0

df_test['BsmtFinSF2'][df_test['BsmtFinSF2'].isnull()] = 0.0

df_test['BsmtUnfSF'][df_test['BsmtUnfSF'].isnull()] = 0.0

df_test['BsmtFullBath'][df_test['BsmtFullBath'].isnull()] = 0.0

df_test['BsmtHalfBath'][df_test['BsmtHalfBath'].isnull()] = 0.0

df_test['KitchenQual'][df_test['KitchenQual'].isnull()] = 'TA'

df_test['Functional'][df_test['Functional'].isnull()] = 'Typ'

df_test['GarageArea'][df_test['GarageArea'].isnull()] = 0.0
for i in true_col_x:

    le = preprocessing.LabelEncoder()

    le.fit(df_test[i])

    df_test[i] = le.transform(df_test[i])
df_test = df_test.drop(false_col_y, axis = 1)
df_test.shape
train = df_train

test = df_test
from sklearn import linear_model #линейная регрессия

from sklearn.ensemble import RandomForestRegressor #rfc регрессор

from sklearn.model_selection import cross_val_score #cross validation

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error #RMSE
X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=80, test_size=.15)
lr = linear_model.LinearRegression()

model_lr = lr.fit(X_train, y_train)

pred_lr = model_lr.predict(X_test)

print ("R^2 is: \n", model_lr.score(X_test, y_test))

print ('RMSE is: \n', mean_squared_error(y_test, pred_lr))
scores = cross_val_score(lr, X_test, y_test, cv = 10)

scores
rf = RandomForestRegressor(n_estimators = 80, random_state = 42, criterion = 'mae')

model_rf = rf.fit(X_train, y_train)

pred_rf = model_rf.predict(X_test)

print ("R^2 is: \n", model_rf.score(X_test, y_test))

print ('RMSE is: \n', mean_squared_error(y_test, pred_rf))
submission = pd.DataFrame()

submission['Id'] = test.Id

feats = test.select_dtypes(

        include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = model_lr.predict(feats)

final_predictions = np.exp(predictions)

print ("Final predictions are: \n", final_predictions[:5])
submission['SalePrice'] = final_predictions

submission.head()