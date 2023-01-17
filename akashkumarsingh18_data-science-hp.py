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
import numpy as np
import pandas as pd
train_data = pd.read_csv('/kaggle/input/train.csv')
test_data = pd.read_csv('/kaggle/input/test.csv')
train_data.head()
train_data.info()
test_data.info()
y = train_data['SalePrice'].values
print(y)
combine_data = pd.concat([train_data.drop(['SalePrice'], axis=1), test_data], axis=0)
print(combine_data)
combine_data.isnull().sum().sum()
combine_data["LotFrontage"] = combine_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
print(combine_data["LotFrontage"])
combine_data["Alley"] = combine_data["Alley"].fillna("None")
print(combine_data["Alley"])
combine_data['MSZoning'] = combine_data['MSZoning'].fillna(combine_data['MSZoning'].mode()[0])
print(combine_data['MSZoning'])
combine_data['Utilities'] = combine_data['Utilities'].fillna(combine_data['Utilities'].mode()[0])
print(combine_data['Utilities'])
combine_data['Exterior1st'] = combine_data['Exterior1st'].fillna(combine_data['Exterior1st'].mode()[0])
print(combine_data['Exterior1st'])
combine_data['Exterior2nd'] = combine_data['Exterior2nd'].fillna(combine_data['Exterior2nd'].mode()[0])
print(combine_data['Exterior2nd'])
combine_data["MasVnrType"] = combine_data["MasVnrType"].fillna(combine_data['MasVnrType'].mode()[0])
print(combine_data["MasVnrType"])
combine_data["MasVnrArea"] = combine_data["MasVnrArea"].fillna(combine_data['MasVnrArea'].mode()[0])
print(combine_data["MasVnrArea"])
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    combine_data[col] = combine_data[col].fillna('None')
print(combine_data[col] )
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    combine_data[col] = combine_data[col].fillna(0)
print(combine_data[col])    
combine_data['Electrical'] = combine_data['Electrical'].fillna(combine_data['Electrical'].mode()[0])
print(combine_data['Electrical'])
combine_data['KitchenQual'] = combine_data['KitchenQual'].fillna(combine_data['KitchenQual'].mode()[0])
print(combine_data['KitchenQual'])
combine_data['Functional'] = combine_data['Functional'].fillna(combine_data['Functional'].mode()[0])
combine_data['FireplaceQu'] = combine_data['FireplaceQu'].fillna('None') 
combine_data['PoolQC'] = combine_data['PoolQC'].fillna('None')
combine_data['Fence'] = combine_data['Fence'].fillna('None')
combine_data['MiscFeature'] = combine_data['MiscFeature'].fillna('None')
combine_data['SaleType'] = combine_data['SaleType'].fillna(combine_data['SaleType'].mode()[0])
print(combine_data['Functional'])
print(combine_data['FireplaceQu'])
print(combine_data['PoolQC'])
print(combine_data['Fence'])
print(combine_data['MiscFeature'])
print(combine_data['SaleType'])
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    combine_data[col] = combine_data[col].fillna('None')
print(combine_data[col])
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    combine_data[col] = combine_data[col].fillna(0)
print(combine_data[col])
combine_data.isnull().sum().sum()
combine_data.info()
combine_data['MSSubClass'] = combine_data['MSSubClass'].astype(str)
print(type(combine_data['MSSubClass']))
combine_data['OverallCond'] = combine_data['OverallCond'].astype(str)
combine_data['OverallQual'] = combine_data['OverallQual'].astype(str)
combine_data = combine_data.drop(['Id'], axis=1)
combine_dummies = pd.get_dummies(combine_data)
print(combine_dummies)
result = combine_dummies.values
print(result)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
result = scaler.fit_transform(result)
X = result[:train_data.shape[0]]
print(X)
test_values = result[train_data.shape[0]:]
print(test_values)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = Lasso()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
from sklearn.metrics import r2_score

print("Train acc: " , r2_score(y_train, y_train_pred))
print("Test acc: ", r2_score(y_test, y_pred))

from sklearn.metrics import mean_squared_error

print("Train acc: " , clf.score(X_train, y_train))
print("Test acc: ", clf.score(X_test, y_test))

final_labels = clf.predict(test_values)
print(final_labels)
#final_result = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': final_labels})
#final_result.to_csv('house_price.csv', index=False)