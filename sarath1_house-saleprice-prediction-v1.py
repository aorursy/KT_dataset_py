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

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn.linear_model as linear_model

import xgboost as xgb

from sklearn import metrics

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from scipy import stats

from IPython.core.interactiveshell import InteractiveShell

from statsmodels.stats.outliers_influence import variance_inflation_factor

InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train_id = train['Id']

test_id = test['Id']

train.set_index('Id', inplace=True)

test.set_index('Id', inplace=True)

train.head()

test.head()

train.shape

test.shape
var_numeric = [f for f in train.columns if train.dtypes[f] != 'object']

var_numeric.remove('SalePrice')

var_categorical = [f for f in train.columns if train.dtypes[f] == 'object']

var_numeric

var_categorical
all_data = pd.concat((train.drop(['SalePrice'],axis=1), test))

all_data.head()
missing_count = all_data.isnull().sum()

missing_count = missing_count[missing_count > 0]

missing_count
fill_none = ['MSZoning','Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC',

            'Fence','MiscFeature']

for col in fill_none:

    all_data[col].fillna('None',inplace=True)

fill_other = ['Exterior1st','Exterior2nd']

for col in fill_other:

    all_data[col].fillna('Other',inplace=True)

all_data['SaleType'].fillna('Oth',inplace=True)

all_data['Electrical'].fillna('SBrkr',inplace=True)

all_data['Functional'].fillna('Typ',inplace=True)

fill_typical = ['KitchenQual']

for col in fill_typical:

    all_data[col].fillna('TA',inplace=True)

fill_zero = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','MasVnrArea',

             'BsmtHalfBath','GarageCars','GarageArea','GarageYrBlt']

for col in fill_zero:

    all_data[col].fillna(0,inplace=True)

all_data['LotFrontage'].fillna(np.nanmedian(all_data.LotFrontage),inplace=True)

missing_count = all_data.isnull().sum()

missing_count = missing_count[missing_count > 0]

missing_count
vif_train_data = all_data.loc[train_id]

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(vif_train_data[var_numeric].values, i) for i in range(len(var_numeric))]

vif["features"] = var_numeric

vif
vif_remove = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF']

var_numeric = [e for e in var_numeric if e not in (vif_remove)]
all_data = all_data[var_numeric+var_categorical]

all_data_encoded = pd.get_dummies(all_data)

df_x_train = all_data_encoded.loc[train_id]

df_y_train = pd.DataFrame(train['SalePrice'])

test_submission = all_data_encoded.loc[test_id]

df_x_train.shape

df_y_train.shape

test_submission.shape
sns.distplot(df_y_train['SalePrice'] , fit=stats.norm)
sm.qqplot(df_y_train['SalePrice'], line='s');
sns.distplot(df_y_train['SalePrice'] , fit=stats.lognorm)
sm.qqplot(np.log(df_y_train["SalePrice"]), line='s');
x_train, x_test, y_train, y_test = train_test_split(df_x_train, df_y_train, test_size=0.33, random_state=42)

x_train.shape

x_test.shape
model = xgb.XGBRegressor()

model.fit(x_train, np.log(y_train))

y_pred = model.predict(x_test)

y_pred = np.exp(y_pred)

y_pred
mse = metrics.mean_squared_error(np.log(y_test), np.log(y_pred))

rmse = np.sqrt(mse)

rmse
model = xgb.XGBRegressor()

model.fit(df_x_train, np.log(df_y_train))

y_pred = model.predict(test_submission)

y_pred = np.exp(y_pred)
test_submission['SalePrice'] = y_pred

test_submission.head()
header = ['SalePrice']

test_submission.to_csv('/kaggle/working/submission_v1.csv', columns = header)