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
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.columns
print(train.shape)

print(test.shape)
train.head(3)
train.isnull().sum()
train = train.drop(['MSZoning','LotFrontage','Alley','Condition1','Condition2','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','BsmtFinType1','BsmtFinType2'], axis = 1)

test = test.drop(['MSZoning','LotFrontage','Alley','Condition1','Condition2','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','BsmtFinType1','BsmtFinType2'], axis = 1)
train = train.drop(['Functional','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','PavedDrive','WoodDeckSF','PoolQC','Fence','MiscFeature','MiscVal'],axis = 1)

test = test.drop(['Functional','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','PavedDrive','WoodDeckSF','PoolQC','Fence','MiscFeature','MiscVal'],axis = 1)
train = train.drop(['BsmtCond','BsmtExposure'], axis = 1)

test = test.drop(['BsmtCond','BsmtExposure'], axis = 1)

train = train.drop(['LandSlope'],axis =1)

test = test.drop(['LandSlope'], axis =1)
train = train.drop(['HeatingQC'], axis = 1)

test = test.drop(['HeatingQC'], axis = 1)
train = train.fillna({'BsmtQual': "TA"})

train = train.fillna({'KitchenQual': 0})

train = train.fillna({'Electrical': "SBrkr"})

train = train.fillna({'GarageQual':"TA"})

train = train.fillna({'GarageCond':"TA"})
train.isnull().sum()
test = test.fillna({'BsmtQual': "TA"})

test = test.fillna({'KitchenQual': 0})

test = test.fillna({'Electrical': "SBrkr"})

test = test.fillna({'GarageQual':"TA",'GarageCond':"TA"})

test = test.fillna({'BsmtFinSF1':0,'BsmtFinSF2':0,'BsmtUnfSF':0})

test = test.fillna({'TotalBsmtSF':0})

test = test.fillna({'Utilities': "AllPub"})

test = test.fillna({'BsmtFullBath':0, 'BsmtHalfBath':1})

test = test.fillna({'GarageCars':2,'GarageArea':1017})

test = test.fillna({'SaleType': "New"})
test.isnull().sum()
train['Street'],_ = pd.factorize(train['Street'])

train['LotShape'],_ = pd.factorize(train['LotShape'])

train['LotConfig'],_ = pd.factorize(train['LotConfig'])

train['LandContour'],_ = pd.factorize(train['LandContour'])

train['Neighborhood'],_ = pd.factorize(train['Neighborhood'])

train['Utilities'],_ = pd.factorize(train['Utilities'])

train['BldgType'],_ = pd.factorize(train['BldgType'])

train['HouseStyle'],_ = pd.factorize(train['HouseStyle'])

train['RoofStyle'],_ = pd.factorize(train['RoofStyle'])

train['ExterQual'],_ = pd.factorize(train['ExterQual'])

train['ExterCond'],_ = pd.factorize(train['ExterCond'])

train['Foundation'],_ = pd.factorize(train['Foundation'])

train['BsmtQual'],_ = pd.factorize(train['BsmtQual'])

train['Heating'],_  = pd.factorize(train['Heating'])

train['CentralAir'],_ = pd.factorize(train['CentralAir'])

train['Electrical'],_ = pd.factorize(train['Electrical'])

train['KitchenQual'],_ = pd.factorize(train['KitchenQual'])

train['GarageQual'],_ = pd.factorize(train['GarageQual'])

train['GarageCond'],_ = pd.factorize(train['GarageCond'])

train['SaleType'],_ = pd.factorize(train['SaleType'])

train['SaleCondition'],_ = pd.factorize(train['SaleCondition'])
test['Street'],_ = pd.factorize(test['Street'])

test['LotShape'],_ = pd.factorize(test['LotShape'])

test['LotConfig'],_ = pd.factorize(test['LotConfig'])

test['LandContour'],_ = pd.factorize(test['LandContour'])

test['Neighborhood'],_ = pd.factorize(test['Neighborhood'])

test['Utilities'],_ = pd.factorize(test['Utilities'])

test['BldgType'],_ = pd.factorize(test['BldgType'])

test['HouseStyle'],_ = pd.factorize(test['HouseStyle'])

test['RoofStyle'],_ = pd.factorize(test['RoofStyle'])

test['ExterQual'],_ = pd.factorize(test['ExterQual'])

test['ExterCond'],_ = pd.factorize(test['ExterCond'])

test['Foundation'],_ = pd.factorize(test['Foundation'])

test['BsmtQual'],_ = pd.factorize(test['BsmtQual'])

test['Heating'],_ = pd.factorize(test["Heating"])

test['CentralAir'],_ = pd.factorize(test['CentralAir'])

test['Electrical'],_ = pd.factorize(test['Electrical'])

test['KitchenQual'],_ = pd.factorize(test['KitchenQual'])

test['GarageQual'],_ = pd.factorize(test['GarageQual'])

test['GarageCond'],_ = pd.factorize(test['GarageCond'])

test['SaleType'],_ = pd.factorize(test['SaleType'])

test['SaleCondition'],_ = pd.factorize(test['SaleCondition'])
import seaborn as sns

graph = sns.countplot(train['BsmtQual'])
train.columns
import sklearn as sk

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import accuracy_score

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

prediction_class = train.drop(['Id','SalePrice'],axis = 1)

target = train['SalePrice']

x_train,x_test,y_train,y_test = train_test_split(prediction_class,target, test_size = 0.33, random_state = 0)
reg = linear_model.LinearRegression()

model = reg.fit(x_train,y_train)

y_predict = reg.predict(x_test)

print(y_predict)
ids = test['Id']

predictions = reg.predict(test.drop(['Id'], axis=1))

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })

output.to_csv('output.csv', index=False)

output_data = pd.read_csv('output.csv')

print(output_data)