# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

import seaborn as sns

import matplotlib.pyplot as plt







add='../input/home-data-for-ml-course/train.csv'

df=pd.read_csv(add)
df.head()
features=['MSSubClass','LotArea','Street','PoolArea','Neighborhood','HouseStyle','YearBuilt','SaleCondition','SaleType','YrSold',

         'OpenPorchSF','GarageArea','Fireplaces','1stFlrSF','2ndFlrSF']

X=df[features]

X.columns=['buildingclass','LotArea','Street','PoolArea','Neighborhood','HouseStyle','YearBuilt','SaleCondition','SaleType','YrSold',

         'OpenPorchSF','GarageArea','Fireplaces','1stFlrSF','2ndFlrSF']

X.head()

y=df['SalePrice']
X.head()
X.info()
X.buildingclass.value_counts()
plt.scatter(X['buildingclass'],y)

#dropping buildingclass

X.drop(['buildingclass'],axis=1,inplace=True)
plt.scatter(X['LotArea'],y)
#dropping rows which has lot area greater than 50000

a=X['LotArea']>45000

i=X[a].index

X.drop(i,inplace=True)

y.drop(i,inplace=True)
plt.scatter(X['LotArea'],y)
X.Street.value_counts()

X.replace(to_replace={'Pave':0,'Grvl':1},inplace=True)
X.PoolArea.value_counts()

plt.scatter(X['PoolArea'],y)

#dropping poolarea

X.drop(['PoolArea'],axis=1,inplace=True)
neighbor=X['Neighborhood'].unique()

neighbor.sort()

print(neighbor)

j=0

for i in neighbor:

    X.replace({i:j},inplace=True)

    j=j+1
plt.scatter(X['Neighborhood'],y)
print(X.HouseStyle.value_counts())

hStyle=X['HouseStyle'].unique()

hStyle.sort()

print(hStyle)

j=0

for i in hStyle:

    X.replace({i:j},inplace=True)

    j=j+1



plt.scatter(X['HouseStyle'],y)    


a=y>700000

Index=y[a].index

y.drop(Index,inplace=True)

X.drop(Index,inplace=True)

plt.scatter(X['YearBuilt'],y)
X.SaleCondition.value_counts()

scondition=X['SaleCondition'].unique()

scondition.sort()

print(scondition)

j=0

for i in scondition:

    X.replace({i:j},inplace=True)

    j=j+1



plt.scatter(X['SaleCondition'],y)    
X.head()
X.SaleType.value_counts()

sType=X['SaleType'].unique()

sType.sort()

print(sType)

j=0

for i in sType:

    X.replace({i:j},inplace=True)

    j=j+1



plt.scatter(X['SaleType'],y)    

#year sold vs price

plt.scatter(X['YrSold'],y)
#year sold vs price

plt.scatter(X['OpenPorchSF'],y)

X.drop(['OpenPorchSF'],axis=1,inplace=True)
plt.scatter(X['GarageArea'],y)

a=X['GarageArea']==0

Index=X[a].index

print(Index)

X.drop(Index,inplace=True)

y.drop(Index,inplace=True)
plt.scatter(X['GarageArea'],y)
plt.scatter(X['Fireplaces'],y)

X.drop(['Fireplaces'],axis=1,inplace=True)
X.head()

plt.scatter(X['1stFlrSF'],y)
plt.scatter(X['2ndFlrSF'],y)

X.drop(['2ndFlrSF'],axis=1,inplace=True)
print(X.head())

features_new=X.columns

print(features_new)
df_test=pd.read_csv("../input/home-data-for-ml-course/test.csv")
df_test_new=df_test[features_new]
df_test_new.head()

df_test_new.info()
df_test_new.replace(to_replace={'Pave':0,'Grvl':1},inplace=True)

neighbor=df_test_new['Neighborhood'].unique()

neighbor.sort()

print(neighbor)

j=0

for i in neighbor:

    df_test_new.replace({i:j},inplace=True)

    j=j+1

    

hStyle=df_test_new['HouseStyle'].unique()

hStyle.sort()

print(hStyle)

j=0

for i in hStyle:

    df_test_new.replace({i:j},inplace=True)

    j=j+1    
scondition=df_test_new['SaleCondition'].unique()

scondition.sort()

print(scondition)

j=0

for i in scondition:

    df_test_new.replace({i:j},inplace=True)

    j=j+1



df_test_new.fillna(value={'SaleType':'WD'},inplace=True)

df_test_new.fillna(value={'GarageArea':472},inplace=True)
sType=df_test_new['SaleType'].unique()

sType.sort()

print(sType)

j=0

for i in sType:

    df_test_new.replace({i:j},inplace=True)

    j=j+1



print(df_test_new.shape)

print(X.shape)

print(y.shape)
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

from sklearn.model_selection import GridSearchCV
dt=DecisionTreeRegressor(random_state=1)

dt.fit(X_train,y_train)

print(dt.score(X_train,y_train))

print(dt.score(X_test,y_test))
param_grid={'max_depth':[2,3,4,5,6,7],'max_leaf_nodes':[5,10,15,20,30,35,40,45,50]}

dt_cv=GridSearchCV(dt,param_grid,cv=5)

dt_cv.fit(X_train,y_train)
import numpy as np

test=dt_cv.cv_results_['mean_test_score']

x=np.arange(54)



plt.plot(x,test)
dt_cv.cv_results_
dt_new=DecisionTreeRegressor(random_state=1,max_depth=7,max_leaf_nodes=30)

dt_new.fit(X_train,y_train)

print(dt_new.score(X_train,y_train))



print(dt_new.score(X_test,y_test))
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(random_state=1,n_estimators=100,max_depth=7,max_leaf_nodes=30)

rf.fit(X_train,y_train)

print(rf.score(X_train,y_train))



print(rf.score(X_test,y_test))

y_pred_rf=rf.predict(df_test_new)

y_pred=dt_new.predict(df_test_new)

y_pred=np.array(y_pred)

df_sam=pd.read_csv("../input/home-data-for-ml-course/sample_submission.csv")
df_sub=pd.DataFrame({'Id':df_sam['Id'],'SalePrice':y_pred})

df_sub_rf=pd.DataFrame({'Id':df_sam['Id'],'SalePrice':y_pred_rf})

df_sub.to_csv("harsh1.csv",index=False)

df_sub_rf.to_csv("harsh_rf.csv",index=False)