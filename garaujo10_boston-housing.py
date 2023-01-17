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
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df.head()
y_train = df.SalePrice.values
df.drop(columns = 'Id', inplace = True)

df_test.drop(columns = 'Id', inplace = True)
trainSize = df.shape[0]

all_data = pd.concat((df, df_test)).reset_index(drop = True)

all_data.drop(columns = "SalePrice", inplace = True)

all_data.shape
import matplotlib.pyplot as plt

import seaborn as sns



plt.hist(y_train)

plt.show()
all_data.columns[all_data.isnull().sum() > 0]   
for i in ('Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'MasVnrType', 'MiscFeature', 'PoolQC'):

    all_data[i].fillna('None', inplace = True)
all_data.columns[all_data.isnull().sum() > 0] 
for h in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[h].fillna(0, inplace = True)
all_data.columns[all_data.isnull().sum() > 0] 
for j in ('GarageArea', 'GarageCars', 'GarageYrBlt', 'LotFrontage', 'MasVnrArea'):

    all_data[j].fillna(0, inplace = True)
all_data.columns[all_data.isnull().sum() > 0] 
for k in ('Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual', 'MSZoning', 'SaleType', 'Utilities'):

    all_data[k].fillna(all_data[j].mode()[0], inplace = True)
all_data.columns[all_data.isnull().sum() > 0] 
factor = all_data.select_dtypes(include = 'object')

numeric = all_data.select_dtypes(include = 'number')
factor.head()
numeric.head()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()
scaler.fit(numeric)
numeric_std = scaler.transform(numeric)
numeric = pd.DataFrame(data = numeric_std, columns = numeric.columns)
numeric.head()
all_data = pd.concat([factor,numeric], axis = 1)
all_data.head()
all_data = pd.get_dummies(all_data)
train = all_data[:trainSize]

test = all_data[trainSize:]
print(train.shape)

print(test.shape)
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
#Creating the models

KNN = KNeighborsRegressor(n_neighbors = 5)

tree = DecisionTreeRegressor()

forest = RandomForestRegressor(n_estimators = 500)
#Training the models

KNN.fit(train, y_train)

tree.fit(train, y_train)

forest.fit(train, y_train)
y_knn = KNN.predict(test)

y_tree = tree.predict(test)

y_forest = forest.predict(test)
ind = np.array(range(1461,2920))
Yy = {'Id': ind, 'SalePrice': y_forest}
pred = pd.DataFrame(data = Yy)
my_submission = pred
my_submission.to_csv('submission.csv', index = False)