# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data  = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
data.head()
data.shape
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline  

plt.figure(figsize = (16,6))

sns.heatmap(data.isnull(),cmap = 'viridis')
plt.figure(figsize = (16,6))

sns.heatmap(test.isnull(),cmap = 'viridis')

data.drop(['Id'], axis = 1,inplace = True)

data.shape

Id = test.Id

test.drop(['Id'], axis = 1,inplace = True)

test.shape
X = data.drop(['SalePrice'],axis = 1)

y = data.SalePrice
test.head()
AgeofHouse = X.YrSold - X.YearRemodAdd

AgeoftestHouse = test.YrSold -test.YearRemodAdd
X = pd.concat([X,AgeofHouse.rename('HouseAge')],axis = 1 )

test = pd.concat([test,AgeoftestHouse.rename('HouseAge')],axis = 1 )
X.drop(['YrSold','YearBuilt','YearRemodAdd'],axis = 1, inplace = True)

test.drop(['YrSold','YearBuilt','YearRemodAdd'],axis = 1, inplace = True)
corr = X.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(corr, cmap='viridis')
X['GarageYrBlt'].corr(X['HouseAge'])
X.columns
X.Fence = X.Fence.fillna('NoFence')

X.MiscFeature = X.MiscFeature.fillna('None')

test.Fence = test.Fence.fillna('NoFence')

test.MiscFeature = test.MiscFeature.fillna('None')
X.PoolQC = X.PoolQC.fillna('NoPool')# description says

test.PoolQC = test.PoolQC.fillna('NoPool')# description says
for col in ['GarageType', 'GarageCond','GarageFinish','GarageQual']:

    X[col] = X[col].fillna('none')

for col in ['GarageType', 'GarageCond','GarageFinish','GarageQual']:

    test[col] = test[col].fillna('none')


X['GarageYrBlt']=X['GarageYrBlt'].fillna(0)

test['GarageYrBlt']=test['GarageYrBlt'].fillna(0)
LF = X.LotFrontage.median()

X['LotFrontage'] = X['LotFrontage'].fillna(LF)

LFt =test.LotFrontage.median()

test['LotFrontage'] = test['LotFrontage'].fillna(LFt)
X['Alley'] = X['Alley'].fillna('none')

X['MasVnrType'] = X['MasVnrType'].fillna('none')

X['MasVnrArea'] = X['MasVnrArea'].fillna(0)

X['GarageCars'] = X['GarageCars'].fillna(0)

X['GarageArea'] = X['GarageArea'].fillna(0)

test['Alley'] = test['Alley'].fillna('none')

test['MasVnrType'] = test['MasVnrType'].fillna('none')

test['MasVnrArea'] = test['MasVnrArea'].fillna(0)

test['GarageCars'] = test['GarageCars'].fillna(0)

test['GarageArea'] = test['GarageArea'].fillna(0)
for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure','BsmtFinType1','BsmtFinType2']:

    X[col] = X[col].fillna('none')

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure','BsmtFinType1','BsmtFinType2']:

    test[col] = test[col].fillna('none')
X['Electrical'] = X['Electrical'].fillna('none')

X['FireplaceQu'] = X['FireplaceQu'].fillna('none')

test['Electrical'] = test['Electrical'].fillna('none')

test['FireplaceQu'] = test['FireplaceQu'].fillna('none')
plt.figure(figsize = (16,6))

sns.heatmap(test.isnull(),cmap = 'viridis')
col_mask=test.isnull().any(axis=0)

col_mask
from sklearn.preprocessing import LabelEncoder
cols = ('MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig',

        'LandSlope','Neighborhood', 'Condition1','Condition2','BldgType','HouseStyle','RoofStyle',

        'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',

        'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',

        'CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish',

        'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition') 
for i in cols:

    

    LE = LabelEncoder() 

    LE.fit(list(X[i].values)) 

    X[i] = LE.transform(list(X[i].values))

   

for i in cols:

    

    LE = LabelEncoder() 

    LE.fit(list(test[i].values)) 

    test[i] = LE.transform(list(test[i].values))

test.head()

from sklearn.linear_model import Ridge
rr = Ridge(alpha=10)

rr.fit(X, y)

y_pred = rr.predict(X)

resid = y - y_pred

mean_resid = resid.mean()

std_resid = resid.std()

z = (resid - mean_resid) / std_resid

z = np.array(z)

outliers = np.where(abs(z) > abs(z).std() * 3)[0]

outliers
X.drop([ 178,  185,  218,  231,  377,  412,  440,  473,  496,  523,  588,

        608,  628,  632,  664,  666,  688,  691,  769,  774,  803,  898,

       1046, 1169, 1181, 1182, 1243, 1298, 1324, 1423] )

y.drop ([ 178,  185,  218,  231,  377,  412,  440,  473,  496,  523,  588,

        608,  628,  632,  664,  666,  688,  691,  769,  774,  803,  898,

       1046, 1169, 1181, 1182, 1243, 1298, 1324, 1423] )      
from sklearn.linear_model import LinearRegression

L1 = LinearRegression()

from sklearn.model_selection import train_test_split

from sklearn import metrics

x_tr,x_ts,y_tr,y_ts = train_test_split(X,y,test_size = 0.3,random_state = 42)
L1.fit(x_tr,y_tr)
pre = L1.predict(x_ts)
metrics.r2_score(pre,y_ts)
plt.scatter(pre,y_ts)
col_mask=test.isnull().any(axis=0)

col_mask
test.info()


col_mask=test.isnull().any(axis=0)

col_mask

test = test.fillna(0)



col_mask=test.isnull().any(axis=0)

col_mask
SalePrice = L1.predict(test)

#test.info()
predict_Sales=pd.Series(SalePrice, name = 'SalePrice')

result = pd.concat([Id, predict_Sales], axis=1)

result.to_csv('Housing_Pred.csv',index=False)

result.info()