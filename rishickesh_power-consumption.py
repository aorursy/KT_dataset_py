# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

import datetime as dt

from keras.models import Sequential

from keras.layers import Dense

import datetime as dt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder



le=LabelEncoder()



from sklearn.preprocessing import scale

train['var2']=le.fit_transform(train['var2'])

test['var2']=le.fit_transform(test['var2'])

train['datetime']=pd.to_datetime(train['datetime']) 

test['datetime']=pd.to_datetime(test['datetime']) 

#train['date']= [d.split() for d in train['datetime']]
def conv(data):

    data['month']=data['datetime'].dt.month

    data['year']=data['datetime'].dt.year

    data['day']=data['datetime'].dt.day

    data['hour']=data['datetime'].dt.hour

    data['sec']=data['datetime'].dt.second

    data['min']=data['datetime'].dt.minute

    

    return data
train=conv(train)

test=conv(test)
train.head()
test.head()
train=train.groupby(['year','month'])[['electricity_consumption']].sum().reset_index()
train['Year month']=train['month'].astype(str)+" "+train["year"].astype(str)
train.tail(7)
X=train.iloc[:, [2,3,4,5,6,8,9,10,11]].values

y=train.iloc[:, 7].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()

mlr.fit(X_train, y_train)



#predict the test set results

y_pred = mlr.predict(X_test)

from sklearn.metrics import mean_squared_error

from math import sqrt



rmse = sqrt(mean_squared_error(y_test, y_pred))



from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



adj_r2 = 1 - float(len(y)-1)/(len(y)-len(mlr.coef_)-1)*(1 - r2)



rmse, r2, adj_r2
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10, weights='distance')

knn.fit(X_train, y_train)



y_pred = knn.predict(X_test)



from sklearn.metrics import mean_squared_error

from math import sqrt

sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



adj_r2 = 1 - float(len(y)-1)/(len(y)-len(mlr.coef_)-1)*(1 - r2)



r2, adj_r2
from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators=20, random_state=0)  

regressor.fit(X_train, y_train)  

y_pred = regressor.predict(X_test) 

from sklearn.metrics import mean_squared_error

from math import sqrt

sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



adj_r2 = 1 - float(len(y)-1)/(len(y)-len(mlr.coef_)-1)*(1 - r2)



r2, adj_r2
from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test) 

from sklearn.metrics import mean_squared_error

from math import sqrt

sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



adj_r2 = 1 - float(len(y)-1)/(len(y)-len(mlr.coef_)-1)*(1 - r2)



r2, adj_r2