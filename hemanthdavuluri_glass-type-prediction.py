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
data = pd.read_csv('/kaggle/input/glass/glass.csv')
data.info()
data.describe()
data.head()
data.corr()
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X=data.drop(columns=['Type'])
y=data.Type

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)
score_train = lr.score(X_train,y_train)
score_test = lr.score(X_test,y_test)

print('Train score : '+str(score_train))
print('Test score : '+str(score_test))
lg=LogisticRegression()
lg.fit(X_train,y_train)
score_train = lg.score(X_train,y_train)
score_test = lg.score(X_test,y_test)

print('Train score : '+str(score_train))
print('Test score : '+str(score_test))
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
score_train = rf.score(X_train,y_train)
score_test = rf.score(X_test,y_test)

print('Train score : '+str(score_train))
print('Test score : '+str(score_test))
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)
score_train = neigh.score(X_train,y_train)
score_test = neigh.score(X_test,y_test)

print('Train score : '+str(score_train))
print('Test score : '+str(score_test))
