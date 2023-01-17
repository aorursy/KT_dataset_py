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
import seaborn as sns 
import pandas as pd 

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train.isnull().sum().sort_values(ascending=False).head(20)
test.info()
train['SalePrice'].describe()
sns.distplot(train['SalePrice'])
train.set_index('Id',inplace = True)

import matplotlib.pyplot as plt 
plt.figure(figsize=(12,12))
sns.heatmap(train.corr(),square=True)
col = train.corr().nlargest(10, 'SalePrice').index
cm = np.corrcoef(train[col].values.T)
sns.heatmap(cm, square=True, cbar=True,annot=True,  xticklabels=col.values, yticklabels=col.values)
col
coll = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',
       'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[coll])
a = train.isnull().sum().sort_values(ascending=False)
b = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
c = pd.concat([a,b])
c.head(25)
sns.scatterplot(x=train['BsmtQual'], y=train['BsmtCond'])
sns.scatterplot(x=train['GarageType'], y=train['GarageCond'])
train['Electrical'].value_counts()
train.dropna(subset=['Electrical'],axis = 0 ,inplace = True)
train.dropna(how='any', axis = 1, inplace = True)
train.isnull().sum().sort_values(ascending=False).head(20)
from scipy.stats import norm
sns.distplot(train['SalePrice'], fit=norm)
test.set_index('Id',inplace = True)
test.info()
test.isnull().sum().sort_values(ascending=False).head(35)
test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageCond', 'GarageFinish', 'GarageYrBlt', 'GarageQual', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MasVnrArea', 'MSZoning'], axis = 1 ,inplace = True)

test = test.fillna(test.mean())
train.drop(['MSZoning'], axis=1, inplace=True)
test = test.apply(lambda x: x.fillna(x.value_counts().index[0]))
test.info()
test = test.apply(lambda x: x.fillna(x.value_counts().index[0]))

train.select_dtypes('object').columns

col = ['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive',
       'SaleType', 'SaleCondition']
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer, OneHotEncoder, StandardScaler
LE = LabelEncoder()
for coli in col:
    train[coli] = LE.fit_transform(train[coli])

LE = LabelEncoder()
for coli in col:
    test[coli] = LE.fit_transform(test[coli])

X_train = train.drop(['SalePrice'], axis=1)
Y_train = train['SalePrice']
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
