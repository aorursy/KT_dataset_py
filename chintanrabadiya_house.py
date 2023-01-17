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
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train.columns
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
df_train.info()
df_train.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)
df_train.shape
df_train['Fence'] = df_train['Fence'].fillna(df_train['Fence'].mode())[0]
df_train.info()
df_train_object = df_train.select_dtypes(include= object)
for i in df_train_object.columns:

    if df_train_object[i].count() < 1460:

        df_train_object[i] = df_train_object.fillna(df_train_object[i].mode()[0])
from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()

for i in df_train_object.columns:

    df_train_object[i] = e.fit_transform(df_train_object[i])
df_train_num = df_train.select_dtypes(include=np.number)
for i in df_train_num.columns:

    if df_train_num[i].count() < 1460:

        df_train_num[i] = df_train_num.fillna(df_train_num[i].mean())
df_train_x = pd.concat([df_train_num, df_train_object], axis=1)

df_train_x.info()
df_test.info()
df_test.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)
df_test_object = df_test.select_dtypes(include=object)
df_test_num = df_test.select_dtypes(include=np.number)
for i in df_test_object.columns:

    if df_test_object[i].count() < 1460:

        df_test_object[i] = df_test_object.fillna(df_test_object[i].mode()[0])
for i in df_test_num.columns:

    if df_test_num[i].count() < 1460:

        df_test_num[i] = df_test_num.fillna(df_test_num[i].mean())
from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()

for i in df_test_object.columns:

    df_test_object[i] = e.fit_transform(df_test_object[i])
df_test_x = pd.concat([df_test_num,df_test_object], axis=1)
df_train_y = df_train_x['SalePrice']

df_train_x.drop(['SalePrice'], axis=1, inplace=True)
x_train = df_train_x[['Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'PavedDrive',

       'SaleCondition', 'Id', 'MSSubClass', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',

       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold']]

x_test = df_test_x[['Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'PavedDrive',

       'SaleCondition', 'Id', 'MSSubClass', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',

       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold']]

       
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(x_train,df_train_y)
pred = model.predict(x_test)
dfs = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

f = {"Id":dfs["Id"],"SalePrice":pred}

f = pd.DataFrame(f)

f.to_csv("submission.csv",index=False)

f.head()