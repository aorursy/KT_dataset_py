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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
h=pd.read_csv("/kaggle/input/predict-the-housing-price/train.csv")

ho1=pd.read_csv("/kaggle/input/predict-the-housing-price/Test.csv")
h.head()
ho1.head()
ID = ho1['Id']
ID
ho1 = ho1.drop(labels = 'Id', axis = 1,inplace = False)
ho1.head()
h.info()
null_columns=h.columns[h.isnull().any()]

hnull=h[null_columns].isnull().sum()

hnull
numerical_feats = h.dtypes[h.dtypes != "object"].index

num=h[numerical_feats].isnull().sum()



categorical_feats = h.dtypes[h.dtypes == "object"].index

cat=h[categorical_feats].isnull().sum()

num
cat
for i in h:

    print(i," ", type(i))

    print(h[i].unique())
cols_fillna = categorical_feats



# replace 'NaN' with 'None' in these columns

for col in cols_fillna:

    h[col].fillna('None',inplace=True)

    ho1[col].fillna('None',inplace=True)
h.fillna(h.mean(),inplace=True)

ho1.fillna(ho1.mean(),inplace=True)
h.info()
ho1.info()
corr=h.corr()

corr["SalePrice"]
plt.figure(figsize=(100,80))

sns.heatmap(h.corr(),annot=True,cmap='RdBu_r')

plt.show()
from sklearn.preprocessing import LabelEncoder



labelencoder = LabelEncoder()

h=h.apply(LabelEncoder().fit_transform)

ho1=ho1.apply(LabelEncoder().fit_transform)
h.head()
ho1.head()
corr=h.corr()
corr["SalePrice"]
h.shape
ho1.shape
x=h[["OverallQual"]]

y=h[["SalePrice"]]

x.shape,y.shape
x.shape,y.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 100)

x_train.shape, x_test.shape
from sklearn.linear_model import LinearRegression

ab = LinearRegression()

ab.fit(x_train, y_train)

ab.intercept_, ab.coef_
from sklearn.metrics import r2_score
y_train_pred = ab.predict(x_train)

r2_score(y_train, y_train_pred)
y_test_pred = ab.predict(x_test)

r2_score(y_test, y_test_pred)
x1_train, x1_test = train_test_split(h, train_size = 0.7, random_state = 100)

x1_train.shape, x1_test.shape
h.columns
y_train = x1_train[['SalePrice']]

x_train =x1_train[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']]

y_test =x1_test[['SalePrice']]

x_test = x1_test[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition',]]
ml1 = LinearRegression()

ml1.fit(x_train, y_train)
ml1.intercept_, ml1.coef_
y_train_pred = ml1.predict(x_train)

r2_score(y_train, y_train_pred)
y_test_pred = ml1.predict(x_test)

r2_score(y_test, y_test_pred)
from sklearn.feature_selection import RFE
rfe = LinearRegression()

rfe_house = RFE(rfe, 35, verbose=True)

rfe_house.fit(x_train, y_train)
rfe_house.support_
colsleft = x_train.columns[rfe_house.support_]

colsleft
reg = LinearRegression()

reg.fit(x_train[colsleft],y_train)
y_train_pred1 = reg.predict(x_train[colsleft])

r2_score(y_train, y_train_pred)
y_test_pred1 = reg.predict(x_test[colsleft])

r2_score(y_test, y_test_pred)
y_test_pred1 = reg.predict(ho1[colsleft])

y_test_pred1

ho1['Id'] = ID
ho1.Id
final = pd.DataFrame(columns=['Id','SalePrice'])

final['Id'] = ho1.Id

final['SalePrice']=y_test_pred1

final
final
final.to_csv('house_Price_predict.csv',index=False)