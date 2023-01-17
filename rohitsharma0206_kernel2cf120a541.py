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
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.drop(['TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea','Id','MSSubClass','MSZoning','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1','Heating','HeatingQC','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','PoolQC','Fence','MiscFeature','MoSold','YrSold','SaleType','SaleCondition'],inplace=True,axis=1)

train.head()

ID=test.Id

test.drop(['Id','MSSubClass','MSZoning','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1','Heating','HeatingQC','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','PoolQC','Fence','MiscFeature','MoSold','YrSold','SaleType','SaleCondition'],inplace=True,axis=1)



train.dropna(inplace=True,axis=1)

test.dropna(inplace=True,axis=1)

test.head()
train.select_dtypes(include=['object'])
train['CentralAir'].loc[(train['CentralAir'] == 'Y')]=1

test['CentralAir'].loc[(test['CentralAir'] == 'Y')]=1
train['CentralAir'].loc[(train['CentralAir'] == 'N')]=0

train['PavedDrive'].loc[(train['PavedDrive'] == 'Y')]=1

train['PavedDrive'].loc[(train['PavedDrive'] == 'N')]=0

train['PavedDrive'].loc[(train['PavedDrive'] == 'P')]=0.5

train['Street'].loc[(train['Street'] == 'Pave')]=1

train['Street'].loc[(train['Street'] == 'Grvl')]=0

test['CentralAir'].loc[(test['CentralAir'] == 'N')]=0

test['PavedDrive'].loc[(test['PavedDrive'] == 'Y')]=1

test['PavedDrive'].loc[(test['PavedDrive'] == 'N')]=0

test['PavedDrive'].loc[(test['PavedDrive'] == 'P')]=0.5

test['Street'].loc[(test['Street'] == 'Pave')]=1

test['Street'].loc[(test['Street'] == 'Grvl')]=0

train.head()
train['ExterCond'].loc[(train['ExterCond'] == 'Ex')]=5

train['ExterCond'].loc[(train['ExterCond'] == 'Gd')]=4

train['ExterCond'].loc[(train['ExterCond'] == 'TA')]=3

train['ExterCond'].loc[(train['ExterCond'] == 'Fa')]=2

train['ExterCond'].loc[(train['ExterCond'] == 'Po')]=1

test['ExterCond'].loc[(test['ExterCond'] == 'Ex')]=5

test['ExterCond'].loc[(test['ExterCond'] == 'Gd')]=4

test['ExterCond'].loc[(test['ExterCond'] == 'TA')]=3

test['ExterCond'].loc[(test['ExterCond'] == 'Fa')]=2

test['ExterCond'].loc[(test['ExterCond'] == 'Po')]=1
train.columns
test.columns
x=train.drop(['SalePrice'],axis=1)

y=pd.DataFrame(train.SalePrice)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,10))

x=pd.DataFrame(scaler.fit_transform(x.values))

y=pd.DataFrame(scaler.fit_transform(y.values))

test=pd.DataFrame(scaler.fit_transform(test.values))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import BayesianRidge

from sklearn.svm import LinearSVR

from sklearn.ensemble import StackingRegressor

estimators = [('lr', BayesianRidge()),('svr', LinearSVR(random_state=42))]

regression = StackingRegressor(estimators=estimators,final_estimator=RandomForestRegressor(n_estimators=10,random_state=42))

reg = Pipeline([('feature_selection', SelectFromModel(LinearRegression())),('regression', regression)])

reg.fit(X_train, y_train)
pred = reg.predict(X_test)

from sklearn.metrics import r2_score

r2_score(y_test,pred)
predictions = reg.predict(test)*100000

submission = pd.DataFrame({'Id':ID,'SalePrice':predictions})

submission                          
filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)