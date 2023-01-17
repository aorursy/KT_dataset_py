# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

sb.set_palette('dark')





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

pid = test.Id
data.head()
print("Missing value count in Training data")

missing_data = data.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print("")
print("Missing value count in Training data")

missing_data1 = test.isnull()

for column in missing_data1.columns.values.tolist():

    print(column)

    print(missing_data1[column].value_counts())

    print("")
data['LotFrontage'] = data['LotFrontage'].replace(np.nan, data['LotFrontage'].mean())

test['LotFrontage'] = test['LotFrontage'].replace(np.nan, test['LotFrontage'].mean())





data['Alley'] = data['Alley'].replace(np.nan,'Grvl')

test['Alley'] = test['Alley'].replace(np.nan,'Grvl')



data['MasVnrArea'] = data['MasVnrArea'].replace(np.nan,data['MasVnrArea'].mean())

test['MasVnrArea'] = test['MasVnrArea'].replace(np.nan,test['MasVnrArea'].mean())







data['BsmtQual'] =  data['BsmtQual'].replace(np.nan,'TA')

test['BsmtQual'] =  test['BsmtQual'].replace(np.nan,'TA')



data['BsmtCond'] =  data['BsmtCond'].replace(np.nan,'TA')

test['BsmtCond'] =  data['BsmtCond'].replace(np.nan,'TA')



data['BsmtExposure'] = data['BsmtExposure'].replace(np.nan,'No')

test['BsmtExposure'] = test['BsmtExposure'].replace(np.nan,'No')



data['BsmtFinType1'] = data['BsmtFinType1'].replace(np.nan,'Unf')

test['BsmtFinType1'] = test['BsmtFinType1'].replace(np.nan,'Unf')



data['BsmtFinType2'] = data['BsmtFinType2'].replace(np.nan,'Unf')

test['BsmtFinType2'] = test['BsmtFinType2'].replace(np.nan,'Unf')



data['Electrical'] = data['Electrical'].replace(np.nan,'SBrkr')



data['GarageType'] = data['GarageType'].replace(np.nan,'Attchd')

test['GarageType'] = test['GarageType'].replace(np.nan,'Attchd')



data['GarageFinish'] = data['GarageFinish'].replace(np.nan,'Unf')

test['GarageFinish'] = test['GarageFinish'].replace(np.nan,'Unf')



data['GarageQual'] = data['GarageQual'].replace(np.nan,'TA')

test['GarageQual'] = test['GarageQual'].replace(np.nan,'TA')





data['GarageCond'] = data['GarageCond'].replace(np.nan,'TA')

test['GarageCond'] = test['GarageCond'].replace(np.nan,'TA')



data['PoolQC'] = data['PoolQC'].replace(np.nan,'Gd')

test['PoolQC'] = test['PoolQC'].replace(np.nan,'Gd')



data['Fence'] = data['Fence'].replace(np.nan,'MnPrv')

test['Fence'] = test['Fence'].replace(np.nan,'MnPrv')



data['MiscFeature'] = data['MiscFeature'].replace(np.nan,'Shed')

test['MiscFeature'] = test['MiscFeature'].replace(np.nan,'Shed')









test['MSZoning'] = test['MSZoning'].fillna('RL')



test['Utilities'] =  test['Utilities'].fillna('ALLPub')



test['Exterior1st'] =  test['Exterior1st'].fillna('VinylSd')



test['Exterior2nd'] =  test['Exterior2nd'].fillna('VinylSd')



test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median()) 



test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].median()) 



test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median())



test ['BsmtFullBath'] = test ['BsmtFullBath'].fillna('0')



test ['HalfBath'] = test ['HalfBath'].fillna('0')



test ['KitchenQual'] = test ['KitchenQual'].fillna('TA')



test ['Functional'] = test ['Functional'].fillna('TA')



test ['GarageCars'] = test ['GarageCars'].fillna(2.0)



test ['GarageArea'] = test ['GarageArea'].fillna(test ['GarageArea'].median())



test['SaleType'] = test['SaleType'].fillna(test['SaleType'].value_counts().idxmax())



test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].value_counts().idxmax())

test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
data.drop(['FireplaceQu','GarageYrBlt','MoSold','YrSold','MasVnrType'],1,inplace=True)

test.drop(['FireplaceQu','GarageYrBlt','MoSold','YrSold','MasVnrType'],1,inplace=True)

data.drop(['Id'],1,inplace=True)

test.drop(['Id'],1,inplace=True)
data.select_dtypes(['object'])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['MSZoning'] = le.fit_transform(data['MSZoning'])

data['Street'] = le.fit_transform(data['Street'])

data['Alley'] = le.fit_transform(data['Alley'])

data['Utilities'] = le.fit_transform(data['Utilities'])

data['LotConfig'] = le.fit_transform(data['LotConfig'])

data['LandSlope'] = le.fit_transform(data['LandSlope'])

data['Neighborhood'] = le.fit_transform(data['Neighborhood'])

data['Condition1'] = le.fit_transform(data['Condition1'])

data['Condition2'] = le.fit_transform(data['Condition2'])

data['GarageType'] = le.fit_transform(data['GarageType'])

data['GarageFinish'] = le.fit_transform(data['GarageFinish'])

data['GarageQual'] = le.fit_transform(data['GarageQual'])

data['GarageCond'] = le.fit_transform(data['GarageCond'])

data['PavedDrive'] = le.fit_transform(data['PavedDrive'])

data['PoolQC'] = le.fit_transform(data['PoolQC'])

data['Fence'] = le.fit_transform(data['Fence'])

data['MiscFeature'] = le.fit_transform(data['MiscFeature'])

data['SaleType'] = le.fit_transform(data['SaleType'])

data['LotShape'] = le.fit_transform(data['LotShape'])

data['LandContour'] = le.fit_transform(data['LandContour'])

data['SaleCondition'] = le.fit_transform(data['SaleCondition'])

data['BldgType'] = le.fit_transform(data['BldgType'])

data['HouseStyle'] = le.fit_transform(data['HouseStyle'])

data['RoofStyle'] = le.fit_transform(data['RoofStyle'])

data['RoofMatl'] = le.fit_transform(data['RoofMatl'])

data['Exterior1st'] = le.fit_transform(data['Exterior1st'])

data['Exterior2nd'] = le.fit_transform(data['Exterior2nd'])

data['ExterQual'] = le.fit_transform(data['ExterQual'])

data['BsmtCond'] = le.fit_transform(data['BsmtCond'])

data['BsmtExposure'] = le.fit_transform(data['BsmtExposure'])

data['BsmtFinType1'] = le.fit_transform(data['BsmtFinType1'])

data['BsmtFinType2'] = le.fit_transform(data['BsmtFinType2'])

data['Heating'] = le.fit_transform(data['Heating'])

data['HeatingQC'] = le.fit_transform(data['HeatingQC'])

data['CentralAir'] = le.fit_transform(data['CentralAir'])

data['Electrical'] = le.fit_transform(data['Electrical'])

data['KitchenQual'] = le.fit_transform(data['KitchenQual'])

data['Functional'] = le.fit_transform(data['Functional'])

data['ExterCond'] = le.fit_transform(data['ExterCond'])

data['Foundation'] = le.fit_transform(data['Foundation'])

data['BsmtQual'] = le.fit_transform(data['BsmtQual'])





test['MSZoning'] = le.fit_transform(test['MSZoning'])

test['Street'] = le.fit_transform(test['Street'])

test['Alley'] = le.fit_transform(test['Alley'])

test['Utilities'] = le.fit_transform(test['Utilities'])

test['LotConfig'] = le.fit_transform(test['LotConfig'])

test['LandSlope'] = le.fit_transform(test['LandSlope'])

test['Neighborhood'] = le.fit_transform(test['Neighborhood'])

test['Condition1'] = le.fit_transform(test['Condition1'])

test['Condition2'] = le.fit_transform(test['Condition2'])

test['GarageType'] = le.fit_transform(test['GarageFinish'])

test['GarageQual'] = le.fit_transform(test['GarageQual'])

test['GarageCond'] = le.fit_transform(test['GarageCond'])

test['PavedDrive'] = le.fit_transform(test['PavedDrive'])

test['PoolQC'] = le.fit_transform(test['PoolQC'])

test['Fence'] = le.fit_transform(test['Fence'])

test['MiscFeature'] = le.fit_transform(test['MiscFeature'])

test['SaleType'] = le.fit_transform(test['SaleType'])

test['LotShape'] = le.fit_transform(test['LotShape'])

test['LandContour'] = le.fit_transform(test['LandContour'])

test['SaleCondition'] = le.fit_transform(test['SaleCondition'])

test['BldgType'] = le.fit_transform(test['BldgType'])

test['HouseStyle'] = le.fit_transform(test['HouseStyle'])

test['RoofStyle'] = le.fit_transform(test['RoofStyle'])

test['RoofMatl'] = le.fit_transform(test['RoofMatl'])

test['Exterior1st'] = le.fit_transform(test['Exterior1st'])

test['Exterior2nd'] = le.fit_transform(test['Exterior2nd'])

test['ExterQual'] = le.fit_transform(test['ExterQual'])

test['BsmtCond'] = le.fit_transform(test['BsmtCond'])

test['BsmtExposure'] = le.fit_transform(test['BsmtExposure'])

test['BsmtFinType1'] = le.fit_transform(test['BsmtFinType1'])

test['BsmtFinType2'] = le.fit_transform(test['BsmtFinType2'])

test['Heating'] = le.fit_transform(test['Heating'])

test['HeatingQC'] = le.fit_transform(test['HeatingQC'])

test['CentralAir'] = le.fit_transform(test['CentralAir'])

test['Electrical'] = le.fit_transform(test['Electrical'])

test['KitchenQual'] = le.fit_transform(test['KitchenQual'])

test['Functional'] = le.fit_transform(test['Functional'])

test['ExterCond'] = le.fit_transform(test['ExterCond'])

test['Foundation'] = le.fit_transform(test['Foundation'])

test['GarageFinish'] = le.fit_transform(test['GarageFinish'])

test['BsmtQual'] = le.fit_transform(test['BsmtQual'])





                                         





data.head()
data.info()
data.describe()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
from sklearn.model_selection import train_test_split,GridSearchCV

y = data['SalePrice']

data.drop('SalePrice',1,inplace=True)

X = data

X = sc.fit_transform(X)

X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
model1 = LinearRegression()

model1.fit(X_train,y_train)
print("Insample Score:",model1.score(X_train,y_train))

print("Outsample Score:",model1.score(X_test,y_test))
model2 = Lasso(alpha=0.001)

model2.fit(X_train,y_train)
print("Insample Score:",model2.score(X_train,y_train))

print("Outsample Score:",model2.score(X_test,y_test))
model3 = Ridge(alpha = 0.1)

model3.fit(X_train,y_train)
print("Insample Score:",model3.score(X_train,y_train))

print("Outsample Score:",model3.score(X_test,y_test))
model4 = ElasticNet(alpha=0.01)

model4.fit(X_train,y_train)
print("Insample Score:",model4.score(X_train,y_train))

print("Outsample Score:",model4.score(X_test,y_test))
param_grid = {'n_neighbors':np.arange(1,9)}

grid_knn = GridSearchCV(KNeighborsRegressor(),param_grid,cv=7)

grid_knn.fit(X,y)
grid_knn.best_params_

model5 = grid_knn.best_estimator_

model5.fit(X_train,y_train)
print("Insample Score:",model5.score(X_train,y_train))

print("Outsample Score:",model5.score(X_test,y_test))
param_grid = {'min_samples_split':np.arange(1,6)}

grid_tree = GridSearchCV(DecisionTreeRegressor(max_depth=6,min_samples_leaf=2),param_grid,cv=7)

grid_tree.fit(X,y)
grid_tree.best_params_
model6 = grid_tree.best_estimator_

model6.fit(X_train,y_train)
print("Insample Score:",model6.score(X_train,y_train))

print("Outsample Score:",model6.score(X_test,y_test))
model8 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=20),n_estimators=1200)

model8.fit(X_train,y_train)
print("Insample Score:",model8.score(X_train,y_train))

print("Outsample Score:",model8.score(X_test,y_test))
from sklearn.ensemble import GradientBoostingRegressor

model6 = GradientBoostingRegressor(max_depth = 8,min_samples_split=3,n_estimators=2000,min_samples_leaf=4)

model6.fit(X_train,y_train)
print("Insample Score:",model6.score(X_train,y_train))

print("Outsample Score:",model6.score(X_test,y_test))
model = GradientBoostingRegressor(max_depth = 8,min_samples_split=3,n_estimators=2000,min_samples_leaf=4)

model.fit(X,y)
prediction = model.predict(test)

prediction
output = pd.DataFrame({'Id':pid,'SalePrice': prediction})

output.to_csv('my_submission1234.csv', index=False)