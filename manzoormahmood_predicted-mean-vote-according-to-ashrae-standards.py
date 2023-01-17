import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_error
data=pd.read_excel('../input/pmv-using-ashrae-standard/HVAC.xlsx',sheet_name=1)

data=data[['Top','Va','RH','met','clo_tot','PMV']]

data.head()
train_data=data[['Top','Va','RH','met','clo_tot']]

train_label=data['PMV']
x_train,x_test,y_train,y_test=train_test_split(train_data,train_label,test_size=0.2,random_state=200)
scaler = StandardScaler()

scaler.fit(x_train) #fit Compute the mean and std to be used for later scaling.

x_train=(scaler.transform(x_train)) #transform-Perform standardization by centering and scaling
scaler = StandardScaler()

scaler.fit(x_test)

x_test=(scaler.transform(x_test))
sgd_reg = SGDRegressor(loss='squared_loss',alpha=0.05, fit_intercept=True, max_iter=200, random_state=123, learning_rate='constant',

                       eta0=0.01)

sgd_reg.fit(x_train, y_train)



y_pred2 = sgd_reg.predict(x_test)

mse = mean_squared_error(y_test,y_pred2)

print((mse))
scaler = StandardScaler()

scaler.fit(train_data) #fit Compute the mean and std to be used for later scaling.

train_data=(scaler.transform(train_data)) #transform-Perform standardization by centering and scaling
x_train,x_test,y_train,y_test=train_test_split(train_data,train_label,test_size=0.1,random_state=200)
sgd_reg = SGDRegressor(loss='squared_loss',alpha=0.005, fit_intercept=True, max_iter=200, random_state=123, learning_rate='constant',

                       eta0=0.01)

sgd_reg.fit(x_train, y_train)



y_pred2 = sgd_reg.predict(x_test)

mse = mean_squared_error(y_test,y_pred2)

print((mse))