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
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train
train = train.drop(['GarageYrBlt'],axis=1)
train = train.drop(['Id'],axis=1)
import seaborn as sns


sns.lineplot(x=train.YrSold,y=train.SalePrice)

sns.boxplot(x=train.YrSold,y=train.SalePrice,hue=train.Fence)
sns.violinplot(x=train.PoolQC,y=train.YrSold,hue=train.CentralAir)
sns.swarmplot(x=train.MSSubClass,y=train.SalePrice,hue=train.HouseStyle)
sns.scatterplot(x = train.SalePrice,y = train.OverallQual,hue = train.OverallCond,style = train.Street)
sns.countplot(x = train.LandContour)
sns.countplot(x = train.HouseStyle)
sns.countplot(x=train.RoofStyle,hue=train.RoofMatl)
train.isnull().sum()
train.LotFrontage.value_counts()
train.LotFrontage.fillna(60.0,inplace=True)
train.isnull().sum()
train.info()
train.Alley.value_counts()
train.Alley.fillna('No alley access',inplace=True)
train. MasVnrType.value_counts() 
train.MasVnrType.fillna('None',inplace=True)
train. MasVnrArea.value_counts() 
train.MasVnrArea.fillna(0.0,inplace=True)
train.BsmtQual.value_counts()
train.BsmtQual.fillna('TA',inplace=True)
train.BsmtCond.value_counts()
train.BsmtExposure.value_counts()
train.BsmtFinType1.value_counts()
train.BsmtFinType2.value_counts()
train.Electrical.value_counts()
train.FireplaceQu.value_counts()
train.GarageType.value_counts()
train.GarageFinish.value_counts()
train.GarageQual.value_counts()
train.GarageCond.value_counts()
train.PoolQC.value_counts()
train.Fence.value_counts()
train.MiscFeature.value_counts()
train.MiscFeature.fillna('None',inplace=True)
train.GarageCond.fillna('TA',inplace=True)
train.Fence.fillna('No Fence',inplace=True)

train.PoolQC.fillna('No Pool',inplace=True)

train.GarageQual.fillna('TA',inplace=True)

train.GarageFinish.fillna('Unf',inplace=True)

train.GarageType.fillna('Attchd',inplace=True)

train.FireplaceQu.fillna('No Fireplace',inplace=True)

train.Electrical.fillna('SBrkr',inplace=True)

train.BsmtFinType1.fillna('Unf',inplace=True)

train.BsmtFinType2.fillna('Unf',inplace=True)

train.BsmtExposure.fillna('No',inplace=True)

train.BsmtCond.fillna('TA',inplace=True)
train.info()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
train['MSZoning']=label.fit_transform(train['MSZoning'])

train['Alley']=label.fit_transform(train['Alley'])

train['LotConfig']=label.fit_transform(train['LotConfig'])

train['LandSlope']=label.fit_transform(train['LandSlope'])

train['LandContour']=label.fit_transform(train['LandContour'])

train['Neighborhood']=label.fit_transform(train['Neighborhood'])

train['Condition1']=label.fit_transform(train['Condition1'])

train['Condition2']=label.fit_transform(train['Condition2'])

train['BldgType']=label.fit_transform(train['BldgType'])

train['HouseStyle']=label.fit_transform(train['HouseStyle'])

train['RoofStyle']=label.fit_transform(train['RoofStyle'])

train['RoofMatl']=label.fit_transform(train['RoofMatl'])

train['Exterior1st']=label.fit_transform(train['Exterior1st'])

train['Exterior2nd']=label.fit_transform(train['Exterior2nd'])

train['ExterQual']=label.fit_transform(train['ExterQual'])

train['ExterCond']=label.fit_transform(train['ExterCond'])

train['Foundation']=label.fit_transform(train['Foundation'])

train['BsmtQual']=label.fit_transform(train['BsmtQual'])

train['BsmtCond']=label.fit_transform(train['BsmtCond'])

train['BsmtExposure']=label.fit_transform(train['BsmtExposure'])

train['BsmtFinSF1']=label.fit_transform(train['BsmtFinSF1'])

train['BsmtFinSF2']=label.fit_transform(train['BsmtFinSF2'])

train['BsmtFinType1']=label.fit_transform(train['BsmtFinType1'])

train['BsmtFinType2']=label.fit_transform(train['BsmtFinType2'])

train['Heating']=label.fit_transform(train['Heating'])

train['HeatingQC']=label.fit_transform(train['HeatingQC'])

train['Electrical']=label.fit_transform(train['Electrical'])

train['KitchenQual']=label.fit_transform(train['KitchenQual'])

train['Functional']=label.fit_transform(train['Functional'])

train['FireplaceQu']=label.fit_transform(train['FireplaceQu'])

train['GarageType']=label.fit_transform(train['GarageType'])

train['GarageCond']=label.fit_transform(train['GarageCond'])

train['GarageFinish']=label.fit_transform(train['GarageFinish'])

train['PavedDrive']=label.fit_transform(train['PavedDrive'])

train['PoolQC']=label.fit_transform(train['PoolQC'])

train['Fence']=label.fit_transform(train['Fence'])

train['MiscFeature']=label.fit_transform(train['MiscFeature'])

train['SaleType']=label.fit_transform(train['SaleType'])

train['SaleCondition']=label.fit_transform(train['SaleCondition'])



one_hot = pd.get_dummies(train['Street'])

train = train.drop('Street',axis=1)

train = train.join(one_hot)
ls = pd.get_dummies(train['LotShape'])

train = train.drop("LotShape",axis=1)

train = train.join(ls)
uls = pd.get_dummies(train['Utilities'])

train = train.drop("Utilities",axis=1)

train = train.join(uls)
mvt = pd.get_dummies(train['MasVnrType'])

train = train.drop("MasVnrType",axis=1)

train = train.join(mvt)
ca = pd.get_dummies(train['CentralAir'])

train = train.drop("CentralAir",axis=1)

train = train.join(ca)
gq = pd.get_dummies(train['GarageQual'])

train = train.drop("GarageQual",axis=1)

train = train.join(gq)
gc = pd.get_dummies(train['GarageCond'])

train = train.drop("GarageCond",axis=1)

train = train.join(gc)
train.info()
def data_splitting(train):

    x=train.drop(['SalePrice'], axis=1)

    y=train['SalePrice']

    return x,y

x_train,y_train = data_splitting(train)

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

log = LinearRegression()

log.fit(x_train,y_train)

log_train = log.score(x_train,y_train)               

print(log_train*100)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test
test = test.drop(['GarageYrBlt'],axis=1)
test = test.drop(['Id'],axis=1)
test.LotFrontage.fillna(60.0,inplace=True)

test.Alley.fillna('No alley access',inplace=True)

test.MasVnrType.fillna('None',inplace=True)

test.MasVnrArea.fillna(0.0,inplace=True)

test.BsmtQual.fillna('TA',inplace=True)

test.MiscFeature.fillna('None',inplace=True)

test.GarageCond.fillna('TA',inplace=True)

test.Fence.fillna('No Fence',inplace=True)

test.PoolQC.fillna('No Pool',inplace=True)

test.GarageQual.fillna('TA',inplace=True)

test.GarageFinish.fillna('Unf',inplace=True)

test.GarageType.fillna('Attchd',inplace=True)

test.FireplaceQu.fillna('No Fireplace',inplace=True)

test.Electrical.fillna('SBrkr',inplace=True)

test.BsmtFinType1.fillna('Unf',inplace=True)

test.BsmtFinType2.fillna('Unf',inplace=True)

test.BsmtExposure.fillna('No',inplace=True)

test.BsmtCond.fillna('TA',inplace=True)
test.info()
test.SaleType.value_counts()
test.SaleType.fillna('WD',inplace=True)
test.MSZoning.value_counts()
test.MSZoning.fillna('RL',inplace=True)
test.Utilities.value_counts()
test.Utilities.fillna('AllPub',inplace=True)
test.Exterior1st.value_counts()
test.Exterior1st.fillna('VinylSd',inplace=True)
test.Exterior2nd.value_counts()
test.Exterior2nd.fillna('VinylSd',inplace=True)
test.BsmtFinSF1.value_counts()
test.BsmtFinSF2.value_counts()
test.TotalBsmtSF.value_counts()
test.BsmtUnfSF.value_counts()
test.BsmtFullBath.value_counts()
test.BsmtHalfBath.value_counts()
test.KitchenQual.value_counts()
test.Functional.value_counts()
test.GarageCars.value_counts()
test.GarageArea.value_counts()
test.GarageArea.fillna(0.0,inplace=True)

test.GarageCars.fillna(2.0,inplace=True)

test.Functional.fillna('typ',inplace=True)

test.KitchenQual.fillna('TA',inplace=True)

test.BsmtHalfBath.fillna(0.0,inplace=True)

test.BsmtFullBath.fillna(0.0,inplace=True)

test.TotalBsmtSF.fillna(0.0,inplace=True)

test.BsmtUnfSF.fillna(0.0,inplace=True)

test.BsmtFinSF2 .fillna(0.0,inplace=True)

test.BsmtFinSF1.fillna(0.0,inplace=True)

test.info()
test['MSZoning']=label.fit_transform(test['MSZoning'])

test['Alley']=label.fit_transform(test['Alley'])

test['LotConfig']=label.fit_transform(test['LotConfig'])

test['LandSlope']=label.fit_transform(test['LandSlope'])

test['LandContour']=label.fit_transform(test['LandContour'])

test['Neighborhood']=label.fit_transform(test['Neighborhood'])

test['Condition1']=label.fit_transform(test['Condition1'])

test['Condition2']=label.fit_transform(test['Condition2'])

test['BldgType']=label.fit_transform(test['BldgType'])

test['HouseStyle']=label.fit_transform(test['HouseStyle'])

test['RoofStyle']=label.fit_transform(test['RoofStyle'])

test['RoofMatl']=label.fit_transform(test['RoofMatl'])

test['Exterior1st']=label.fit_transform(test['Exterior1st'])

test['Exterior2nd']=label.fit_transform(test['Exterior2nd'])

test['ExterQual']=label.fit_transform(test['ExterQual'])

test['ExterCond']=label.fit_transform(test['ExterCond'])

test['Foundation']=label.fit_transform(test['Foundation'])

test['BsmtQual']=label.fit_transform(test['BsmtQual'])

test['BsmtCond']=label.fit_transform(test['BsmtCond'])

test['BsmtExposure']=label.fit_transform(test['BsmtExposure'])

test['BsmtFinSF1']=label.fit_transform(test['BsmtFinSF1'])

test['BsmtFinSF2']=label.fit_transform(test['BsmtFinSF2'])

test['BsmtFinType1']=label.fit_transform(test['BsmtFinType1'])

test['BsmtFinType2']=label.fit_transform(test['BsmtFinType2'])

test['Heating']=label.fit_transform(test['Heating'])

test['HeatingQC']=label.fit_transform(test['HeatingQC'])

test['Electrical']=label.fit_transform(test['Electrical'])

test['KitchenQual']=label.fit_transform(test['KitchenQual'])

test['Functional']=label.fit_transform(test['Functional'])

test['FireplaceQu']=label.fit_transform(test['FireplaceQu'])

test['GarageType']=label.fit_transform(test['GarageType'])

test['GarageCond']=label.fit_transform(test['GarageCond'])

test['GarageFinish']=label.fit_transform(test['GarageFinish'])

test['PavedDrive']=label.fit_transform(test['PavedDrive'])

test['PoolQC']=label.fit_transform(test['PoolQC'])

test['Fence']=label.fit_transform(test['Fence'])

test['MiscFeature']=label.fit_transform(test['MiscFeature'])

test['SaleType']=label.fit_transform(test['SaleType'])

test['SaleCondition']=label.fit_transform(test['SaleCondition'])
gq = pd.get_dummies(test['GarageQual'])

test = test.drop("GarageQual",axis=1)

test = test.join(gq)

ca = pd.get_dummies(test['CentralAir'])

test = test.drop("CentralAir",axis=1)

test = test.join(ca)

mvt = pd.get_dummies(test['MasVnrType'])

test = test.drop("MasVnrType",axis=1)

test = test.join(mvt)

uls = pd.get_dummies(test['Utilities'])

test = test.drop("Utilities",axis=1)

test = test.join(uls)

ls = pd.get_dummies(test['LotShape'])

test = test.drop("LotShape",axis=1)

test = test.join(ls)

one_hot = pd.get_dummies(test['Street'])

test = test.drop('Street',axis=1)

test = test.join(one_hot)

gc = pd.get_dummies(test['GarageCond'])

test = test.drop("GarageCond",axis=1)

test = test.join(gc)
test.info()