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

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline

sns.set_style('whitegrid')

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_data.head()
#train_data.info()

sns.heatmap(train_data.isnull())
train_data= train_data.drop(['Alley','PoolQC','FireplaceQu','Fence','MiscFeature'], axis=1)
sns.heatmap(train_data.iloc[:,:].isnull())
train_data['LotFrontage']= train_data[['LotFrontage']].fillna(value=train_data['LotFrontage'].mean())



garageType=pd.get_dummies(train_data['GarageType'], drop_first=True)

train_data.drop(['GarageType'], axis=1,inplace=True)

train_data=pd.concat([train_data,garageType], axis=1)



train_data['GarageYrBlt']= train_data[['GarageYrBlt']].fillna(value=train_data['GarageYrBlt'].mean())



GarageFinish=pd.get_dummies(train_data['GarageFinish'], drop_first=True)

train_data.drop(['GarageFinish'], axis=1,inplace=True)

train_data=pd.concat([train_data,GarageFinish], axis=1)



Electrical=pd.get_dummies(train_data['Electrical'], drop_first=True)

train_data.drop(['Electrical'], axis=1,inplace=True)

train_data=pd.concat([train_data,Electrical], axis=1)



train_data['BsmtFinSF2']= train_data[['BsmtFinSF2']].fillna(value=train_data['BsmtFinSF2'].mean())



BsmtQual=pd.get_dummies(train_data['BsmtQual'], drop_first=True)

train_data.drop(['BsmtQual'], axis=1,inplace=True)

train_data=pd.concat([train_data,BsmtQual], axis=1)



train_data.drop(['GarageCond'], axis=1,inplace=True)



GarageQual=pd.get_dummies(train_data['GarageQual'], drop_first=True)

train_data.drop(['GarageQual'], axis=1,inplace=True)

train_data=pd.concat([train_data,GarageQual], axis=1)



train_data['MasVnrArea']= train_data[['MasVnrArea']].fillna(value=train_data['MasVnrArea'].mean())



MasVnrType=pd.get_dummies(train_data['MasVnrType'], drop_first=True)

train_data.drop(['MasVnrType'], axis=1,inplace=True)

train_data=pd.concat([train_data,MasVnrType], axis=1)



train_data.drop(['BsmtCond','BsmtFinType1','BsmtFinType2'], axis=1,inplace=True)

test_data['BsmtFullBath']= test_data[['BsmtFullBath']].fillna(value=test_data['BsmtFullBath'].mean())

test_data['BsmtHalfBath']= test_data[['BsmtHalfBath']].fillna(value=test_data['BsmtHalfBath'].mean())



BsmtExposure=pd.get_dummies(train_data['BsmtExposure'], drop_first=True)

train_data.drop(['BsmtExposure'], axis=1,inplace=True)

train_data=pd.concat([train_data,BsmtExposure], axis=1)



MSZoning=pd.get_dummies(train_data['MSZoning'], drop_first=True)

train_data.drop(['MSZoning'], axis=1,inplace=True)

train_data=pd.concat([train_data,MSZoning], axis=1)



train_data.drop(['Street'], axis=1,inplace=True)



LotShape=pd.get_dummies(train_data['LotShape'], drop_first=True)

train_data.drop(['LotShape'], axis=1,inplace=True)

train_data=pd.concat([train_data,LotShape], axis=1)



LandContour=pd.get_dummies(train_data['LandContour'], drop_first=True)

train_data.drop(['LandContour'], axis=1,inplace=True)

train_data=pd.concat([train_data,LandContour], axis=1)



train_data.drop(['Utilities'], axis=1,inplace=True)



LotConfig=pd.get_dummies(train_data['LotConfig'], drop_first=True)

train_data.drop(['LotConfig'], axis=1,inplace=True)

train_data=pd.concat([train_data,LotConfig], axis=1)



HouseStyle=pd.get_dummies(train_data['HouseStyle'], drop_first=True)

train_data.drop(['HouseStyle'], axis=1,inplace=True)

train_data=pd.concat([train_data,HouseStyle], axis=1)



Foundation=pd.get_dummies(train_data['Foundation'], drop_first=True)

train_data.drop(['Foundation'], axis=1,inplace=True)

train_data=pd.concat([train_data,Foundation], axis=1)



CentralAir=pd.get_dummies(train_data['CentralAir'], drop_first=True)

train_data.drop(['CentralAir'], axis=1,inplace=True)

train_data=pd.concat([train_data,CentralAir], axis=1)



SaleType=pd.get_dummies(train_data['SaleType'], drop_first=True)

train_data.drop(['SaleType'], axis=1,inplace=True)

train_data=pd.concat([train_data,SaleType], axis=1)



SaleCondition=pd.get_dummies(train_data['SaleCondition'], drop_first=True)

train_data.drop(['SaleCondition'], axis=1,inplace=True)

train_data=pd.concat([train_data,SaleCondition], axis=1)
sns.heatmap(train_data.iloc[:,:].isnull())
from sklearn.model_selection import train_test_split

X=train_data.drop(['SalePrice','Id','LandSlope',

                   'Neighborhood','Condition1','Condition2','BldgType','RoofStyle','RoofMatl',

                   'Exterior1st','Exterior2nd','ExterQual','ExterCond','Heating','HeatingQC',

                   'KitchenQual','Functional','PavedDrive','Mix','IR2','IR3','Reg','Y','Fa','Gd','2.5Fin'], axis=1)

y=train_data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30,random_state=501)



X

# print(y)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



rf = RandomForestClassifier(n_estimators=800)

rf.fit(X_train,y_train)

predict_with_rf=rf.predict(X_test)



rf_val_mae = mean_absolute_error(predict_with_rf,y_test)

print(rf_val_mae)

X_test
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sns.heatmap(test_data.iloc[:,:].isnull())
test_data
test_data= test_data.drop(['Alley','PoolQC','FireplaceQu','Fence','MiscFeature'], axis=1)
test_data['LotFrontage']= test_data[['LotFrontage']].fillna(value=test_data['LotFrontage'].mean())



garageType=pd.get_dummies(test_data['GarageType'], drop_first=True)

test_data.drop(['GarageType'], axis=1,inplace=True)

test_data=pd.concat([test_data,garageType], axis=1)



test_data['GarageYrBlt']= test_data[['GarageYrBlt']].fillna(value=test_data['GarageYrBlt'].mean())



GarageFinish=pd.get_dummies(test_data['GarageFinish'], drop_first=True)

test_data.drop(['GarageFinish'], axis=1,inplace=True)

test_data=pd.concat([test_data,GarageFinish], axis=1)



Electrical=pd.get_dummies(test_data['Electrical'], drop_first=True)

test_data.drop(['Electrical'], axis=1,inplace=True)

test_data=pd.concat([test_data,Electrical], axis=1)



test_data['BsmtFinSF2']= test_data[['BsmtFinSF2']].fillna(value=test_data['BsmtFinSF2'].mean())



BsmtQual=pd.get_dummies(test_data['BsmtQual'], drop_first=True)

test_data.drop(['BsmtQual'], axis=1,inplace=True)

test_data=pd.concat([test_data,BsmtQual], axis=1)



test_data.drop(['GarageCond'], axis=1,inplace=True)



GarageQual=pd.get_dummies(test_data['GarageQual'], drop_first=True)

test_data.drop(['GarageQual'], axis=1,inplace=True)

test_data=pd.concat([test_data,GarageQual], axis=1)



test_data['MasVnrArea']= test_data[['MasVnrArea']].fillna(value=test_data['MasVnrArea'].mean())

test_data['BsmtFullBath']= test_data[['BsmtFullBath']].fillna(value=test_data['BsmtFullBath'].mean())

test_data['BsmtHalfBath']= test_data[['BsmtHalfBath']].fillna(value=test_data['BsmtHalfBath'].mean())



MasVnrType=pd.get_dummies(test_data['MasVnrType'], drop_first=True)

test_data.drop(['MasVnrType'], axis=1,inplace=True)

test_data=pd.concat([test_data,MasVnrType], axis=1)



test_data.drop(['BsmtCond','BsmtFinType1','BsmtFinType2'], axis=1,inplace=True)



BsmtExposure=pd.get_dummies(test_data['BsmtExposure'], drop_first=True)

test_data.drop(['BsmtExposure'], axis=1,inplace=True)

test_data=pd.concat([test_data,BsmtExposure], axis=1)



MSZoning=pd.get_dummies(test_data['MSZoning'], drop_first=True)

test_data.drop(['MSZoning'], axis=1,inplace=True)

test_data=pd.concat([test_data,MSZoning], axis=1)



test_data.drop(['Street'], axis=1,inplace=True)



LotShape=pd.get_dummies(test_data['LotShape'], drop_first=True)

test_data.drop(['LotShape'], axis=1,inplace=True)

train_data=pd.concat([test_data,LotShape], axis=1)



LandContour=pd.get_dummies(train_data['LandContour'], drop_first=True)

test_data.drop(['LandContour'], axis=1,inplace=True)

test_data=pd.concat([test_data,LandContour], axis=1)



test_data.drop(['Utilities'], axis=1,inplace=True)



LotConfig=pd.get_dummies(test_data['LotConfig'], drop_first=True)

test_data.drop(['LotConfig'], axis=1,inplace=True)

test_data=pd.concat([test_data,LotConfig], axis=1)



HouseStyle=pd.get_dummies(test_data['HouseStyle'], drop_first=True)

test_data.drop(['HouseStyle'], axis=1,inplace=True)

test_data=pd.concat([test_data,HouseStyle], axis=1)



Foundation=pd.get_dummies(test_data['Foundation'], drop_first=True)

test_data.drop(['Foundation'], axis=1,inplace=True)

test_data=pd.concat([test_data,Foundation], axis=1)



CentralAir=pd.get_dummies(test_data['CentralAir'], drop_first=True)

test_data.drop(['CentralAir'], axis=1,inplace=True)

train_data=pd.concat([test_data,CentralAir], axis=1)



SaleType=pd.get_dummies(test_data['SaleType'], drop_first=True)

test_data.drop(['SaleType'], axis=1,inplace=True)

test_data=pd.concat([test_data,SaleType], axis=1)



SaleCondition=pd.get_dummies(test_data['SaleCondition'], drop_first=True)

test_data.drop(['SaleCondition'], axis=1,inplace=True)

test_data=pd.concat([test_data,SaleCondition], axis=1)
sns.heatmap(test_data.iloc[:,:].isnull())
X_test_data=test_data.drop(['Id','LandSlope',

                   'Neighborhood','Condition1','Condition2','BldgType','RoofStyle','RoofMatl',

                   'Exterior1st','Exterior2nd','ExterQual','ExterCond','Heating','HeatingQC',

                   'KitchenQual','Functional','PavedDrive','Fa','Gd'], axis=1)



sns.heatmap(X_test_data.iloc[:,:].isnull())

for column in X_test_data:

    X_test_data[column]= X_test_data[[column]].fillna(value=X_test_data[column].mean())

X_test_data

X_test.info()
X_test_data.info()
y_pred=rf.predict(X_test_data)

# rf_val_mae = mean_absolute_error(predict_with_rf,y_test)

# print(rf_val_mae)

y_pred

submission= pd.DataFrame({ 

    'Id': test_data['Id'],

    'SalePrice': y_pred })

print(submission)

submission.to_csv("Submission_house_price.csv", index=False)