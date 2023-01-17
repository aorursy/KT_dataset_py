# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()
data.shape
data.isnull().sum()
data.fillna({'reviews_per_month':0}, inplace=True)
data.fillna({'name':"NoName"}, inplace=True)
data.fillna({'host_name':"NoName"}, inplace=True)
data.fillna({'last_review':"NotReviewed"}, inplace=True)
data.isnull().sum()
data.columns
data.dtypes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data.drop(['name','id','host_name','host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month','last_review'], axis=1, inplace=True)
data.head(5)
data_en=data.copy() 
for column in data.columns[data.columns.isin(['neighbourhood_group', 'room_type'])]:
    data_en[column] = data[column].factorize()[0]
    
data_en.head(5)
lm = LinearRegression()

X = data_en.iloc[:,[0,1,3,4,5]]
y=data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

lm.fit(X_train,y_train)
y_pred=lm.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.4,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
from sklearn.svm import SVR
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.1,random_state=105)
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)
#for i in range(len(x_test)-1)
y_predict=regressor.predict(x_test)
r2_score(y_test,y_predict)
from sklearn.svm import SVR
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.4,random_state=105)
regressor = SVR(kernel = 'sigmoid')
regressor.fit(x_train, y_train)
y_predict=regressor.predict(x_test)
r2_score(y_test,y_predict)
