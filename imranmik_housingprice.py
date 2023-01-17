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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from matplotlib.artist import setp
train = pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()

train.columns
train = train.drop(['Condition2','Foundation','FireplaceQu','PavedDrive','MiscFeature','Fence','PoolQC','Alley'],axis=1)

test = test.drop(['Condition2','Foundation','FireplaceQu','PavedDrive','MiscFeature','Fence','PoolQC','Alley'],axis=1)
train.head()
train.shape
train= train.fillna(method='ffill')

test= test.fillna(method='ffill')
train.isnull().any()
plt.figure(figsize=(20,10))

sns.countplot(data=train,x='MSSubClass')

plt.figure(figsize=(20,10))

sns.countplot(data=train,x='YearBuilt')

plt.xticks(rotation=90)
plt.figure(figsize=(20,15))

sns.heatmap(train.corr())
sns.countplot(data=train, x='OverallQual')
plt.figure(figsize=(20,10))

sns.distplot(train['SalePrice'],color ='b')
train.head()
train['MSZoning'].unique()

MSZoning_map={'RL':0,'RM':1,'C(all)':2,'FV':3,'RH':4}

for data in train,test:

    data['MSZoning']=data['MSZoning'].map(MSZoning_map)

    data['MSZoning']=data['MSZoning'].fillna(0)

train['Street'].unique()

street_map={'Pave':0,'Grvl':1}

for data in train,test:

    data['Street']=data['Street'].map(street_map)
train['LotShape'].unique()

Lotshape_map={'Reg':0,'IR1':1,'IR2':2,'IR3':3}

for data in test,train:

    data['LotShape']= data['LotShape'].map(Lotshape_map)

train['LandContour'].unique()

landcontour_map={'Lvl':0,'Bnk':1,'Low':2,'HLS':3}

for data in train,test:

    data['LandContour']= data['LandContour'].map(landcontour_map)

    
train.head()
attributes_train = ['SalePrice','Street','LotShape','LandContour', 'MSSubClass', 'MSZoning', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'LotArea', 'GarageCars', 'GarageArea', 'EnclosedPorch']

attributes_test =  ['MSSubClass', 'Street','LotShape','LandContour','MSZoning', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'LotArea', 'GarageCars', 'GarageArea', 'EnclosedPorch']

train = train[attributes_train]

test1 =test[attributes_test]
X=train.drop(['SalePrice'],axis=1)

y=train['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
model = GradientBoostingRegressor(n_estimators=1500,learning_rate=0.01,max_depth=5)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
a=model.score(X_test, y_test)

print('The score using GradientBoostingRegressor is ',round(a*100,2))
model1=RandomForestRegressor(n_estimators=1500,max_features='sqrt',max_depth=5)

model1.fit(X_train,y_train)

y_pred=model1.predict(X_test)
a1=model1.score(X_test,y_test)

print('The score using RandomForestRegressor is ',round(a1*100,2))
predictions = model.predict(test1)
submission=pd.DataFrame({'Id':test['Id'],'SalePrice':predictions})

submission.to_csv('submission.csv',index=False)