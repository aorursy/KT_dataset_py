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
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
'''

Data fields

Here's a brief version of what you'll find in the data description file.



SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.

MSSubClass: The building class

MSZoning: The general zoning classification

LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access

Alley: Type of alley access

LotShape: General shape of property

LandContour: Flatness of the property

Utilities: Type of utilities available

LotConfig: Lot configuration

LandSlope: Slope of property

Neighborhood: Physical locations within Ames city limits

Condition1: Proximity to main road or railroad

Condition2: Proximity to main road or railroad (if a second is present)

BldgType: Type of dwelling

HouseStyle: Style of dwelling

OverallQual: Overall material and finish quality

OverallCond: Overall condition rating

YearBuilt: Original construction date

YearRemodAdd: Remodel date

RoofStyle: Type of roof

RoofMatl: Roof material

Exterior1st: Exterior covering on house

Exterior2nd: Exterior covering on house (if more than one material)

MasVnrType: Masonry veneer type

MasVnrArea: Masonry veneer area in square feet

ExterQual: Exterior material quality

ExterCond: Present condition of the material on the exterior

Foundation: Type of foundation

BsmtQual: Height of the basement

BsmtCond: General condition of the basement

BsmtExposure: Walkout or garden level basement walls

BsmtFinType1: Quality of basement finished area

BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Quality of second finished area (if present)

BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating

HeatingQC: Heating quality and condition

CentralAir: Central air conditioning

Electrical: Electrical system

1stFlrSF: First Floor square feet

2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Number of bedrooms above basement level

Kitchen: Number of kitchens

KitchenQual: Kitchen quality

TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality rating

Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

GarageType: Garage location

GarageYrBlt: Year garage was built

GarageFinish: Interior finish of the garage

GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

GarageCond: Garage condition

PavedDrive: Paved driveway

WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality

Fence: Fence quality

MiscFeature: Miscellaneous feature not covered in other categories

MiscVal: $Value of miscellaneous feature

MoSold: Month Sold

YrSold: Year Sold

SaleType: Type of sale

SaleCondition: Condition of sale

'''
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train = train.set_index('Id')

test = test.set_index('Id')
train.head()
test.head()
print('Dataset Size: ')

print('Treino: ', train.shape)

print('Test: ', test.shape)
columns = test.columns

columns
print('Percentage of NA values per variable in train: ')

for i in range(len(columns)):

    value = train[columns[i]].isna().sum() / 1460 * 100

    if value > 0.0:

        print('{}: {:.2f}%'.format(columns[i], value))
print('Percentage of NA values per variable in test: ')

for i in range(len(columns)):

    value = test[columns[i]].isna().sum() / 1459 * 100

    if value > 0.0:

        print('{}: {:.2f}%'.format(columns[i], value))
del train['Alley']

del train['PoolQC']

del train['MiscFeature']

del test['Alley']

del test['PoolQC']

del test['MiscFeature']
train = train.fillna(train.mode().iloc[0])

test = test.fillna(test.mode().iloc[0])

train.dtypes
test.dtypes
columns = test.columns
y_train = train['SalePrice']

del train['SalePrice']



dtypes = test.dtypes



for i in range(train.shape[1]):

    if (train[columns[i]].dtype == test[columns[i]].dtype) == False:

        train[columns[i]] = train[columns[i]].astype(test[columns[i]].dtype)
for i in range(train.shape[1]):

    print(train[columns[i]].dtype == test[columns[i]].dtype)
train = pd.get_dummies(train)

test = pd.get_dummies(test)
train.head()
test.head()
print('New number of variables in Train: ',train.shape[1])

print('New number of variables in Test: ',test.shape[1])
traincol = train.columns
for i in range(len(traincol)):

    if traincol[i] not in test:

        test[traincol[i]] = pd.Series([0], dtype=train[traincol[i]].dtype)

        test[traincol[i]] = 0

        print('{} type: {}'.format(traincol[i], train[traincol[i]].dtype))
print('After matching: ')

print('New number of variables in Train: ',train.shape[1])

print('New number of variables in Test: ',test.shape[1])
train.dtypes
train.dtypes
'''

from sklearn.preprocessing import Normalizer



normalizer = Normalizer().fit(train)



train['SalePrice'] = y_train

xtrainnorm = pd.DataFrame(normalizer.transform(train), columns=train.columns, index=train.index)

xtestvalnorm = pd.DataFrame(normalizer.transform(test), columns=test.columns, index=test.index)

y_train = xtrainnorm['SalePrice']

del xtrainnorm['SalePrice']

'''



train['SalePrice'] = y_train

xtrainnorm = np.log1p(train)

xtestvalnorm = np.log1p(test)

y_train = xtrainnorm['SalePrice']

del xtrainnorm['SalePrice']





# rearranging, because xgb raises an error if the columns are in different order

xtestvalnorm = xtestvalnorm.reindex(sorted(xtestvalnorm.columns), axis=1)

xtrainnorm = xtrainnorm.reindex(sorted(xtrainnorm.columns), axis=1)
xtrainnorm.head()
xtestvalnorm.head()
from sklearn.model_selection import train_test_split

X_train, X_test, yy_train, y_test = train_test_split(xtrainnorm, y_train, test_size=0.33, random_state=0)
X_train.head()
X_test.head()
yy_train.head()
y_test.head()
import xgboost as xgb

from sklearn.metrics import mean_squared_error

model = xgb.XGBRegressor(learning_rate=0.01, n_estimators=4110,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:squarederror', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)

 

model.fit(X_train, yy_train)



# predict the target on the train dataset

predict_train = model.predict(X_train)

 

# RMSE Score on train dataset

rmse_train = np.sqrt(mean_squared_error(yy_train, predict_train))

print('\nRMSE on train dataset : ', rmse_train)

 

# predict the target on the test dataset

predict_test = model.predict(X_test)

 

# RMSE Score on test dataset

rmse_test = np.sqrt(mean_squared_error(y_test, predict_test))

print('\nRMSE on test dataset : ', rmse_test)
model.fit(xtrainnorm, y_train)



predicted_prices = np.expm1(model.predict(xtestvalnorm))

submission = pd.DataFrame({'Id': test.index, 'SalePrice': predicted_prices})

submission.index += 1 

submission.to_csv('submission.csv', index=False)