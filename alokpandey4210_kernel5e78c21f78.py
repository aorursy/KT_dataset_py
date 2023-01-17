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
df=pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

df.head()
df.corr(method='pearson')
X=df.drop(['longitude','total_bedrooms','population','median_house_value'],axis=1)
X.columns.tolist()
X.isnull().sum()
y=df['median_house_value']
X=pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.40)
from sklearn.linear_model import LinearRegression

y=LinearRegression()

y.fit(X_train,y_train)

y.score(X_test,y_test)
from sklearn.tree import DecisionTreeRegressor

h=DecisionTreeRegressor(max_depth=3)

h.fit(X_train,y_train)

h.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor

l=RandomForestRegressor()

l.fit(X_train,y_train)

l.score(X_test,y_test)
from sklearn.linear_model import SGDRegressor

qa=SGDRegressor()

qa.fit(X_train,y_train)

qa.score(X_test,y_test)

from sklearn.ensemble import ExtraTreesRegressor

wz=ExtraTreesRegressor()

wz.fit(X_train,y_train)

wz.score(X_test,y_test)


    