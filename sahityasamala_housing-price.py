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

import matplotlib.pyplot as plt
train = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"

test = "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"

train = pd.read_csv(train)

test = pd.read_csv(test)

train.head()
train.tail()
train.describe()
train.info()
train.isna().sum()
import seaborn  as sns

sns.heatmap(train.isnull(),cbar=False)
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train = train.drop(['Alley'],axis = 1)
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])

train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])

train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])

train['BsmtFinType1'] = train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0])
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])

train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])

train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])

train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())

train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])

train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train = train.drop(['PoolQC','Fence','MiscFeature'],axis = 1)
train = train.drop(['Id'],axis = 1)
sns.heatmap(train.isnull(),cbar=False)
test.head()
test.isna().sum()
test.info()

sns.heatmap(test.isnull(),cbar=False)
#filling the missing values in the test dataset

test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())

test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])

test['BsmtCond']=test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])

test['BsmtQual']=test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])



test['FireplaceQu']=test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])

test['GarageType']=test['GarageType'].fillna(test['GarageType'].mode()[0])



test['GarageFinish']=test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])

test['GarageQual']=test['GarageQual'].fillna(test['GarageQual'].mode()[0])



test['GarageCond']=test['GarageCond'].fillna(test['GarageCond'].mode()[0])

test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())



test['BsmtExposure']=test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])

test['BsmtFinType2']=test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])





test['Utilities']=test['Utilities'].fillna(test['Utilities'].mode()[0])

test['Exterior1st']=test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['Exterior2nd']=test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

test['BsmtFinType1']=test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])

test['BsmtFinSF1']=test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['BsmtFinSF2']=test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())

test['BsmtUnfSF']=test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())

test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())

test['BsmtFullBath']=test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])

test['BsmtHalfBath']=test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])

test['KitchenQual']=test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['Functional']=test['Functional'].fillna(test['Functional'].mode()[0])

test['GarageCars']=test['GarageCars'].fillna(test['GarageCars'].mean())

test['GarageArea']=test['GarageArea'].fillna(test['GarageArea'].mean())

test['SaleType']=test['SaleType'].fillna(test['SaleType'].mode()[0])

test = test.drop(['Alley','PoolQC'],axis=1)

test = test.drop(['Fence','MiscFeature'],axis =1)
sns.heatmap(test.isnull(),cbar=False)
y = train["SalePrice"]
X_train = train.drop(['SalePrice'],axis = 1)
X_train.columns
X_test = test

X_test.columns
X_train=train.select_dtypes(include='object')

X_train.head(20)
#Encode categorical features as an integer array.



from sklearn.preprocessing import OrdinalEncoder 

oe=OrdinalEncoder()

enX_train=oe.fit_transform(X_train)

enX_train
enX_train2=pd.DataFrame(data=enX_train,columns=X_train.columns)

enX_train2
num_train=train.select_dtypes(include='int')

num_train
float_train=train.select_dtypes(include='float')



float_train
final_train=pd.concat([enX_train2,num_train,float_train],axis=1)

final_train.head(10)
X_test=test.select_dtypes(include='object')

X_test.head(20)
X_test=X_test.fillna('dummy')
#Encode categorical features as an integer array.



from sklearn.preprocessing import OrdinalEncoder 

oe=OrdinalEncoder()

enX_test=oe.fit_transform(X_test)

enX_test
enX_test2=pd.DataFrame(data=enX_test,columns=X_test.columns)

enX_test2
num_test=test.select_dtypes(include='int')

num_test
float_test=test.select_dtypes(include='float')



float_test
final_test=pd.concat([enX_test2,num_test,float_test],axis=1)

final_test
final_train=final_train.fillna(0)

final_test=final_test.fillna(0)
TRAIN = pd.DataFrame(data=final_train) 

TEST = pd.DataFrame(data=final_test)
type(TRAIN)
type(TEST)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(TRAIN)

X_test = sc.fit_transform(TEST)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X_train, y)
y_pred = regressor.predict(X_test)

print(y_pred)
Output = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})

Output.to_csv('My_submission.csv', index=False)

print(Output)