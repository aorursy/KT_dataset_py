#import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')





to_drop = ['Id','MoSold','YrSold','Street','Condition1','Condition2','YearRemodAdd','Exterior1st','Exterior2nd',

          'BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

           'SaleType']

train.drop(to_drop, axis=1, inplace=True)

test.drop(to_drop, axis=1, inplace=True)





train = pd.concat([train.drop(['MSZoning'],axis=1), pd.get_dummies(train[['MSZoning']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['SaleCondition'],axis=1), pd.get_dummies(train[['SaleCondition']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['LotShape'],axis=1), pd.get_dummies(train[['LotShape']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['Utilities'],axis=1), pd.get_dummies(train[['Utilities']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['LandContour'],axis=1), pd.get_dummies(train[['LandContour']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['LandSlope'],axis=1), pd.get_dummies(train[['LandSlope']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['RoofStyle'],axis=1), pd.get_dummies(train[['RoofStyle']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['Heating'],axis=1), pd.get_dummies(train[['Heating']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['CentralAir'],axis=1), pd.get_dummies(train[['CentralAir']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['LotConfig'],axis=1), pd.get_dummies(train[['LotConfig']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['HouseStyle'],axis=1), pd.get_dummies(train[['HouseStyle']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['Foundation'],axis=1), pd.get_dummies(train[['Foundation']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['Neighborhood'],axis=1), pd.get_dummies(train[['Neighborhood']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['HeatingQC'],axis=1), pd.get_dummies(train[['HeatingQC']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['Electrical'],axis=1), pd.get_dummies(train[['Electrical']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['GarageType'],axis=1), pd.get_dummies(train[['GarageType']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['GarageFinish'],axis=1), pd.get_dummies(train[['GarageFinish']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['KitchenQual'],axis=1), pd.get_dummies(train[['KitchenQual']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['OverallQual'],axis=1), pd.get_dummies(train[['OverallQual']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['ExterQual'],axis=1), pd.get_dummies(train[['ExterQual']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['BsmtQual'],axis=1), pd.get_dummies(train[['BsmtQual']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['Functional'],axis=1), pd.get_dummies(train[['Functional']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['MSSubClass'],axis=1), pd.get_dummies(train[['MSSubClass']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['BldgType'],axis=1), pd.get_dummies(train[['BldgType']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['MasVnrType'],axis=1), pd.get_dummies(train[['MasVnrType']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['PavedDrive'],axis=1), pd.get_dummies(train[['PavedDrive']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['GarageQual'],axis=1), pd.get_dummies(train[['GarageQual']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['GarageCond'],axis=1), pd.get_dummies(train[['GarageCond']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['ExterCond'],axis=1), pd.get_dummies(train[['ExterCond']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['BsmtCond'],axis=1), pd.get_dummies(train[['BsmtCond']], drop_first=True)], axis=1)

train = pd.concat([train.drop(['RoofMatl'],axis=1), pd.get_dummies(train[['RoofMatl']], drop_first=True)], axis=1)

train = train.loc[:,train.isnull().sum()/len(train)<=0.3]





test = pd.concat([test.drop(['MSZoning'],axis=1), pd.get_dummies(test[['MSZoning']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['SaleCondition'],axis=1), pd.get_dummies(test[['SaleCondition']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['LotShape'],axis=1), pd.get_dummies(test[['LotShape']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['Utilities'],axis=1), pd.get_dummies(test[['Utilities']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['LandContour'],axis=1), pd.get_dummies(test[['LandContour']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['LandSlope'],axis=1), pd.get_dummies(test[['LandSlope']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['RoofStyle'],axis=1), pd.get_dummies(test[['RoofStyle']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['Heating'],axis=1), pd.get_dummies(test[['Heating']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['CentralAir'],axis=1), pd.get_dummies(test[['CentralAir']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['LotConfig'],axis=1), pd.get_dummies(test[['LotConfig']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['HouseStyle'],axis=1), pd.get_dummies(test[['HouseStyle']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['Foundation'],axis=1), pd.get_dummies(test[['Foundation']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['Neighborhood'],axis=1), pd.get_dummies(test[['Neighborhood']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['HeatingQC'],axis=1), pd.get_dummies(test[['HeatingQC']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['Electrical'],axis=1), pd.get_dummies(test[['Electrical']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['GarageType'],axis=1), pd.get_dummies(test[['GarageType']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['GarageFinish'],axis=1), pd.get_dummies(test[['GarageFinish']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['KitchenQual'],axis=1), pd.get_dummies(test[['KitchenQual']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['OverallQual'],axis=1), pd.get_dummies(test[['OverallQual']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['ExterQual'],axis=1), pd.get_dummies(test[['ExterQual']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['BsmtQual'],axis=1), pd.get_dummies(test[['BsmtQual']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['Functional'],axis=1), pd.get_dummies(test[['Functional']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['MSSubClass'],axis=1), pd.get_dummies(test[['MSSubClass']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['BldgType'],axis=1), pd.get_dummies(test[['BldgType']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['MasVnrType'],axis=1), pd.get_dummies(test[['MasVnrType']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['PavedDrive'],axis=1), pd.get_dummies(test[['PavedDrive']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['GarageQual'],axis=1), pd.get_dummies(test[['GarageQual']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['GarageCond'],axis=1), pd.get_dummies(test[['GarageCond']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['ExterCond'],axis=1), pd.get_dummies(test[['ExterCond']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['BsmtCond'],axis=1), pd.get_dummies(test[['BsmtCond']], drop_first=True)], axis=1)

test = pd.concat([test.drop(['RoofMatl'],axis=1), pd.get_dummies(test[['RoofMatl']], drop_first=True)], axis=1)



features = [f for f in test.columns if f in train.columns]



train.shape
X = SimpleImputer().fit_transform(train[features])

X_predict = SimpleImputer().fit_transform(test[features])

y = train['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X,y)
rfc = RandomForestRegressor().fit(X_train, y_train)

rfc.score(X_test,y_test)
y_predicted = rfc.predict(X_predict)

submission = pd.DataFrame({'Id':test.index,'SalePrice':y_predicted})

submission.head()
submission.to_csv('Predicted_Prices.csv',index=False)