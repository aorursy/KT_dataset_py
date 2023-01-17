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
df=pd.read_csv('/kaggle/input/tesla-stock-price/Tesla.csv - Tesla.csv.csv')

df.head()
df.corr(method='pearson')
X=df.drop(['Date','Adj Close'],axis=1)
X.head()
y=df['Adj Close']
X.isnull().sum()
X=pd.get_dummies(X)
X['Open'].hist(bins=50)
X.boxplot(column='Open')
X['Open'].plot('density',color='Red')
X['High'].hist(bins=50)
X.boxplot(column='High')
X['High'].plot('density',color='Yellow')
X['Low'].hist(bins=50)
X.boxplot(column='Low')
X['Low'].plot('density',color='Yellow')
X['Close'].hist(bins=50)
X.boxplot(column='Close')
X['Close'].plot('density',color='Black')
X['Volume'].hist(bins=50)
X.boxplot(column='Volume')
X['Volume'].plot('density',color='Green')
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.tree import DecisionTreeRegressor

q=DecisionTreeRegressor()

q.fit(X_train,y_train)

q.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor

w=RandomForestRegressor()

w.fit(X_train,y_train)

w.score(X_test,y_test)
from sklearn.linear_model import LinearRegression

d=LinearRegression()

d.fit(X_train,y_train)

d.score(X_test,y_test)
from sklearn.linear_model import SGDRegressor

t=SGDRegressor()

t.fit(X_train,y_train)

t.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesRegressor

b=ExtraTreesRegressor()

b.fit(X_train,y_train)

b.score(X_test,y_test)
#save model

import pickle 

file_name='Stock_Price.sav'

tuples=(d,X)

pickle.dump(tuples,open(file_name,'wb'))
