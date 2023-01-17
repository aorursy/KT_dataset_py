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
df=pd.read_csv('/kaggle/input/black-friday/train.csv')

df.head()
df.corr(method='pearson')
X=df.drop(['User_ID','Product_ID','Occupation','Marital_Status','Product_Category_3','Purchase'],axis=1)

X.columns.tolist()
y=df['Purchase']
X.isnull().sum()
X['Product_Category_2'].value_counts()
X['Product_Category_2'].fillna(X['Product_Category_2'].mean(),inplace=True)
X=pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
from sklearn.linear_model import LinearRegression

f=LinearRegression()

f.fit(X_train,y_train)

f.score(X_test,y_test)
from sklearn.tree import DecisionTreeRegressor

t=DecisionTreeRegressor()

t.fit(X_train,y_train)

t.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor

n=RandomForestRegressor()

n.fit(X_train,y_train)

n.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesRegressor

g=ExtraTreesRegressor()

g.fit(X_train,y_train)

g.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesRegressor

m=ExtraTreesRegressor()

m.fit(X_train,y_train)

m.score(X_test,y_test)
#save model

import pickle

file_name='Friday.sav'

tuples=(n,X)

pickle.dump(tuples,open(file_name,'wb'))
yqw=n.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error

re1=mean_squared_error(y_test,yqw)

re2=mean_absolute_error(y_test,yqw)

print(re1,' ',re2)