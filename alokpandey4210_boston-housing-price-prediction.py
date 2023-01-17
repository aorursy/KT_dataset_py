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
df=pd.read_csv('/kaggle/input/bostonhoustingmlnd/housing.csv')

df.head()
df.corr(method='pearson')
X=df.drop(['MEDV'],axis=1)
X.head()
y=df['MEDV']
X.isnull().sum()
X['RM'].hist(bins=50)
X.hist(bins=50)
X.boxplot(column='RM',by='LSTAT')
X['RM'].plot('density',color='Red')
X['LSTAT'].plot('density',color='Green')
X.boxplot(column='LSTAT',by='PTRATIO')
X['PTRATIO'].plot('density',color='Blue')
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)
from sklearn.tree import DecisionTreeRegressor

q=DecisionTreeRegressor()

q.fit(X_train,y_train)

q.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor

w=RandomForestRegressor()

w.fit(X_train,y_train)

w.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesRegressor

c=ExtraTreesRegressor()

c.fit(X_train,y_train)

c.score(X_test,y_test)
from sklearn.linear_model import RANSACRegressor

o=RANSACRegressor()

o.fit(X_train,y_train)

o.score(X_test,y_test)
from sklearn.linear_model import LinearRegression

l=LinearRegression()

l.fit(X_train,y_train)

l.score(X_test,y_test)
#save model

import pickle

file_name='Boston.sav'

tuples=(c,X)

pickle.dump(tuples,open(file_name,'wb'))
yqv=c.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error

m1=mean_squared_error(y_test,yqv)

m2=mean_absolute_error(y_test,yqv)

print(m1,'  ',m2)