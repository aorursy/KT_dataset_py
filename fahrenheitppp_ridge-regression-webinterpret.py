import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt 



house = pd.read_csv("../input/train.csv", index_col=0)

test = pd.read_csv("../input/test.csv", index_col=0)

fulldata = pd.concat((house, test), axis=0)

house.shape, test.shape, fulldata.shape
plt.figure(figsize=[10,10])

sns.heatmap(house.corr(), square=True)

plt.show()
missdata = fulldata.isnull().sum().sort_values(ascending=False)

missdata.head(40)
fulldata = fulldata.drop(['PoolQC','MiscFeature','Alley','Fence'],1)

fulldata.shape
fulldata["MasVnrArea"] = fulldata["MasVnrArea"].fillna(0)

fulldata["MasVnrType"] = fulldata["MasVnrType"].fillna('None')
fulldata['LotFrontage'] = fulldata['LotFrontage'].fillna(fulldata['LotFrontage'].mean())
basement = ['BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond', 'BsmtFinType1']

garage = ['GarageQual', 'GarageFinish','GarageCond','GarageType']

basement.extend(garage)

for column in basement:

    fulldata[column] = fulldata[column].fillna('None')  
others = ['BsmtFullBath','BsmtHalfBath','TotalBsmtSF','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF','GarageCars','GarageArea','GarageYrBlt']

for col in others:

    fulldata[col] = fulldata[col].fillna(0)
fulldata["Electrical"] = fulldata["Electrical"].fillna('SBrkr')

fulldata["FireplaceQu"] = fulldata["FireplaceQu"].fillna('None')

fulldata["MSZoning"] = fulldata["MSZoning"].fillna('RL')

fulldata["Functional"] = fulldata["Functional"].fillna('Typ')

fulldata["SaleType"] = fulldata["SaleType"].fillna('WD')

fulldata["Utilities"] = fulldata["Utilities"].fillna('AllPub')

fulldata["KitchenQual"] = fulldata["KitchenQual"].fillna('TA')

fulldata["Exterior1st"] = fulldata["Exterior1st"].fillna('VinylSD')

fulldata["Exterior2nd"] = fulldata["Exterior2nd"].fillna('VinylSD')
fulldata.isnull().sum().sort_values(ascending=False).head()
fullnum = pd.get_dummies(fulldata)

fullnum.shape
train = fullnum.loc[house.index]

x_train = train.drop('SalePrice',1)

y_train = train['SalePrice']

x_test = fullnum.loc[test.index].drop('SalePrice',1)
from sklearn import linear_model



reg = linear_model.RidgeCV(alphas=np.linspace(0.1,20,100))

reg.fit(x_train, y_train)

reg.alpha_
ridge = linear_model.Ridge(alpha=10.75)

ridge.fit(x_train,y_train)

my_submission = pd.DataFrame(data={'id':test.index, 'SalePrice':ridge.predict(x_test)}, columns=('id','SalePrice'))

my_submission = my_submission.set_index('id')

my_submission.to_csv('submissionRidge.csv')