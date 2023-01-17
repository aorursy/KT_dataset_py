# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression 

from sklearn.linear_model import LogisticRegression





features=pd.read_csv('/kaggle/input/features.csv')

train1=pd.read_csv('/kaggle/input/train.csv')



stores=pd.read_csv('/kaggle/input/stores.csv')

stores.head()





train2 = pd.merge(features, train1, on=['Store','Date','IsHoliday'], how='inner')



train=pd.merge(train2,stores,how='inner',on=['Store'])

train.drop(columns=['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'], axis=1, inplace=True)



train.dropna(inplace=True)



X=train[['Store','Temperature','Fuel_Price','CPI','Unemployment','Dept','Size']]

y=train['Weekly_Sales']



X_train,X_test,y_train,y_test=train_test_split( X, y, test_size=0.25, random_state=42)
rfr = RandomForestRegressor(n_estimators = 250, random_state=0)               #RandomForestRegressor

rfr.fit(X_train,y_train)

y_pred=rfr.predict(X_test)
print(metrics.mean_absolute_error(y_test, y_pred))

print(metrics.mean_squared_error(y_test, y_pred))

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

accuracy = rfr.score(X_test,y_test)

print('accuracy RandomForestRegressor: ',accuracy*100,'%')