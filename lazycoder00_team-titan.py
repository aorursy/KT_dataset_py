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
import pandas as pd
import numpy as np
train=pd.read_csv('../input/walmart-sales/Train.csv')

test=pd.read_csv('../input/walmart-sales/Test.csv')
train.head(5)
train.info()
import seaborn as sns

import matplotlib.pyplot as plt
sns.countplot(x='Item_Fat_Content',data=train)

sns.countplot(x='Outlet_Size',data=train)

sns.countplot(x='Outlet_Location_Type',data=train)



sns.countplot(x='Outlet_Size',hue='Outlet_Location_Type',data=train)

ax=sns.countplot(x='Outlet_Type',data=train)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)



ax=sns.countplot(x='Item_Type',data=train)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)





plot=train['Outlet_Establishment_Year'].plot.hist()

plot.set_xlabel('Outlet_Establishment_Year')
plot=train['Item_MRP'].plot.hist()

plot.set_xlabel('Item_MRP')
sns.heatmap(train.isnull(),yticklabels='False')

sns.heatmap(train.corr(),yticklabels='True')

train.isnull().sum()
train1=pd.get_dummies(train["Item_Fat_Content"])

test1=pd.get_dummies(test["Item_Fat_Content"])

train1
train3=pd.get_dummies(train["Outlet_Type"])

test3=pd.get_dummies(test["Outlet_Type"])



train3
train2=pd.get_dummies(train["Outlet_Location_Type"])

test2=pd.get_dummies(test["Outlet_Location_Type"])



train2
train.drop(["Item_Fat_Content","Outlet_Location_Type","Outlet_Type",'Item_Weight','Outlet_Size'],axis=1,inplace=True)

test.drop(["Item_Fat_Content","Outlet_Location_Type","Outlet_Type",'Item_Weight','Outlet_Size'],axis=1,inplace=True)
train.head(2)
sns.heatmap(train.isnull(),yticklabels='False')
train_data=pd.concat([train,train1,train2,train3],axis=1)

train_data.head(5)
test_data=pd.concat([test,test1,test2,test3],axis=1)

test_data.head(3)
train_data.head(5)
train_data.isnull().sum()
train_data['Item_Outlet_Sales']=train_data['Item_Outlet_Sales'].astype('int64')

x=pd.get_dummies(train_data['Item_Type'])
train_data=pd.concat([x,train_data],axis=1,)

train_data.drop('Item_Type',axis=1,inplace=True)
train_data['Low Fat']=train_data['Low Fat']+ train_data['LF']+ train_data['low fat']
train_data['Regular']=train_data['Regular']+ train_data['reg']

train_data.drop('reg',axis=1,inplace=True)

train_data.drop(['LF','low fat'],axis=1,inplace=True)

train_data.dtypes
(train_data['Item_Outlet_Sales'].max(),train_data['Item_Outlet_Sales'].min())

train_data.shape
for i in range(1,6):

    name= 'Item_Outlet_Sales' +str(i)

    train_data[name]= (2500*(i-2) <= train_data['Item_Outlet_Sales']) & (train_data['Item_Outlet_Sales']<=2500*(i-1))

    train_data[name]=train_data[name].astype('int64')

    train_data[name]=train_data[name].replace(1,i-1)

train_data.shape
train_data['Item_Outlet_Sales_discrete']=train_data[train_data.columns[31:36]].sum(axis=1,skipna=True)
train_data.drop(train_data.iloc[:,31:36],axis=1,inplace=True)
train_data.drop('Item_Outlet_Sales',axis=1,inplace=True)
train_data.head(4)
(train_data['Item_MRP'].min(),train_data['Item_MRP'].max())
train_data.shape
for i in range(1,6):

    name= 'Item_MRP' +str(i)

    train_data[name]= (46*(i-2) <= train_data['Item_MRP']) & (train_data['Item_MRP']<=46*(i-1))

    train_data[name]=train_data[name].astype('int64')

    train_data[name]=train_data[name].replace(1,i-1)

train_data.shape
train_data['Item_MRP_net']=train_data[train_data.columns[31:36]].sum(axis=1,skipna=True)
train_data.shape
train_data.drop(train_data.iloc[:,31:36],axis=1,inplace=True)
train_data.head(4)
train_data.drop('Item_MRP',axis=1,inplace=True)
train_data.columns
X=train_data.drop(['Item_Identifier','Item_Visibility','Outlet_Identifier','Item_Outlet_Sales_discrete'],axis=1)

y=train_data['Item_Outlet_Sales_discrete']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn import neighbors

from sklearn.tree import DecisionTreeClassifier
LR=LogisticRegression()

GB=GaussianNB()

KNN=neighbors.KNeighborsClassifier(n_neighbors=8)

DTC=DecisionTreeClassifier()
LR.fit(X_train,y_train)
GB.fit(X_train,y_train)

KNN.fit(X_train,y_train)

DTC.fit(X_train,y_train)

y_predict1=LR.predict(X_test)

y_predict2=GB.predict(X_test)

y_predict3=KNN.predict(X_test)

y_predict4=DTC.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score1=accuracy_score(y_test,y_predict1)

accuracy_score2=accuracy_score(y_test,y_predict2)

accuracy_score3=accuracy_score(y_test,y_predict3)

accuracy_score4=accuracy_score(y_test,y_predict4)
(accuracy_score1,accuracy_score2,accuracy_score3,accuracy_score4)