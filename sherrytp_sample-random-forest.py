# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/petrol-consumption/petrol_consumption.csv') 
data.head()
X = data.iloc[:, 0:4].values  

y = data.iloc[:, 4].values  
# import sklearn 

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)  
# Feature Scaling

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()  

X_train = sc.fit_transform(X_train)  

X_test = sc.transform(X_test)  
from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators=20, random_state=0)  

regressor.fit(X_train, y_train)  

y_pred = regressor.predict(X_test)  
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators=100, random_state=0)  

regressor.fit(X_train, y_train)  

y_pred = regressor.predict(X_test)  
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
data = pd.read_csv('../input/bank-note-authentication-uci-data/BankNote_Authentication.csv')
data.head()
X = data.iloc[:, 0:4].values  

y = data.iloc[:, 4].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  
sc = StandardScaler()  

X_train = sc.fit_transform(X_train)  

X_test = sc.transform(X_test)  
from sklearn.ensemble import RandomForestClassifier 

rf_model = RandomForestClassifier(n_estimators=20, random_state=0)  

rf_model.fit(X_train, y_train)  

y_pred = rf_model.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(confusion_matrix(y_test,y_pred))  

print(classification_report(y_test,y_pred))  

print(accuracy_score(y_test, y_pred))  
rf_model2 = RandomForestClassifier(n_estimators=100, random_state=0)  

rf_model2.fit(X_train, y_train)  

y_pred = rf_model2.predict(X_test)  
print(confusion_matrix(y_test,y_pred))  

print(classification_report(y_test,y_pred))  

print(accuracy_score(y_test, y_pred))  
# alternative way to write the feature and label dataset 

X = data.drop('class', axis = 1)

y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

clt = DecisionTreeClassifier()

clt.fit(X_train, y_train)

y_pred = clt.predict(X_test)
print(confusion_matrix(y_test,y_pred))  

print(classification_report(y_test,y_pred))  

print(accuracy_score(y_test, y_pred))  