import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv('../input/big-mart-sales-pred/train_Av.csv',delimiter=',')

test=pd.read_csv('../input/big-mart-sales-pred/test_Av.csv',delimiter=',')
ID=test['Item_Identifier']

OD=test['Outlet_Identifier']
train.head()
test.info()
train['Item_Identifier']


new_year=train['Outlet_Establishment_Year'].map(lambda x:(2013-x))
train['Outlet_Establishment_Year']=new_year
train['Outlet_Establishment_Year']
train.Outlet_Identifier.unique()
train.drop(['Item_Identifier'],axis=1,inplace=True)

test.drop(['Item_Identifier'],axis=1,inplace=True)
train.info()
sns.countplot(x='Item_Fat_Content',data=train)
sns.boxplot(x='Item_Weight',data=train)
sns.countplot(x='Outlet_Size',hue='Outlet_Location_Type',data=train)
train['Item_Weight']=train['Item_Weight'].fillna(train.Item_Weight.mean())

test['Item_Weight']=test['Item_Weight'].fillna(test.Item_Weight.mean())
train['Outlet_Size']=train['Outlet_Size'].fillna(train.Outlet_Size.mode()[0])

test['Outlet_Size']=test['Outlet_Size'].fillna(test.Outlet_Size.mode()[0])
train.isnull().sum().sort_values()
train_objs_num = len(train)

dataset = pd.concat(objs=[train, test], axis=0)

dataset = pd.get_dummies(dataset)

 

dataset
train=dataset.iloc[:train_objs_num]
test=dataset.iloc[train_objs_num:]
train
x=train.drop(['Item_Outlet_Sales'],axis=1)

y=train['Item_Outlet_Sales']
x
min_train=x.min()
range_train=(x-min_train).max()
x_scaled=(x-min_train)/range_train
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=.25)
x_train.shape
from xgboost.sklearn import XGBRFRegressor

xgb=XGBRFRegressor()

xgb.fit(x_train,y_train)
y_pred_xg=xgb.predict(x_test)
xgb.score(x_test,y_test)
test.shape
test.drop(['Item_Outlet_Sales'],axis=1,inplace=True)
test.head()
new_year_test=test['Outlet_Establishment_Year'].map(lambda x:(2013-x))
test['Outlet_Establishment_Year']=new_year_test
test['Outlet_Establishment_Year']
min_test=test.min()
range_test=(test-min_test).max()
test_scaled=(test-min_test)/range_test
test_scaled
y_pred_xg=xgb.predict(test_scaled)
y_pred_xg
output=pd.DataFrame({'Item_Identifier':ID,'Outlet_Identifier':OD,'Item_Outlet_Sales':y_pred_xg})

output.to_csv('submission_xg1.csv',index=None)