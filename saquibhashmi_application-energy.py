import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.metrics import classification_report,accuracy_score,mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import xgboost as xgb

import seaborn as sbn
os.chdir('../input')
os.listdir()
data = pd.read_csv('energydata_complete.csv')
data.shape
data.columns
data.dtypes
data.isnull().sum()
data['date']
data['date'] = pd.to_timedelta(data['date'].astype('datetime64')).dt.seconds
data['date'].dtype
y = data['Appliances']
X = data

del X['Appliances']
X.columns
X.shape
pd.plotting.scatter_matrix(data.iloc[:,:7],figsize = (10,10))
X = X.drop(['T2','RH_2','T3'],axis=1)
X.shape
X['date'].describe()
plt.plot(X['date'])
plt.scatter(np.arange(len(X['date'])),X['date'])
X = X.drop(['date'],axis=1)
X.shape
plt.hist(X['lights'])
X['lights'].describe()
pd.plotting.scatter_matrix(data.iloc[:,14:23],figsize = (10,10))
X = X.drop(['T7','T8','RH_7','RH_8'],axis=1)
X.shape
pd.plotting.scatter_matrix(data.iloc[:,23:28],figsize=(10,10))
del X['rv1']
plt.scatter(np.arange(len(X['rv2'])),X['rv2'])
del X['rv2']
X.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
X_train.shape,X_test.shape
X.dtypes
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train.shape
def adjusted_r_square(n,k,y,yhat):

    SS_Residual = sum((y-yhat)**2)

    SS_Total = sum((y-np.mean(y))**2)

    r_squared = 1 - (float(SS_Residual))/SS_Total

    adjusted_r_squared = 1 - (((1-r_squared)*(k-1))/(n-k-1))

    return r_squared,adjusted_r_squared
linReg = LinearRegression()

linReg.fit(X_train,y_train)
linReg.score(X_train,y_train)
y_predLinReg = linReg.predict(scaler.transform(X_test))
adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predLinReg)
mean_squared_error(y_predLinReg,y_test)
DecisonReg = DecisionTreeRegressor()

DecisonReg
DecisonReg.fit(X_train,y_train)

DecisonReg.score(X_train,y_train)
y_predDecisonReg = DecisonReg.predict(scaler.transform(X_test))



mean_squared_error(y_predDecisonReg,y_test)
adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predDecisonReg)
DecisonReg.feature_importances_
X.columns
RandomForestReg = RandomForestRegressor(n_estimators=600)

RandomForestReg
RandomForestReg.fit(X_train,y_train)

RandomForestReg.score(X_train,y_train)
y_predRandomForestReg = RandomForestReg.predict(scaler.transform(X_test))



mean_squared_error(y_predRandomForestReg,y_test)
adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predRandomForestReg)
svmReg = SVR(C = 0.75)

svmReg
svmReg.fit(X_train,y_train)

svmReg.score(X_train,y_train)
y_predsvmReg = svmReg.predict(scaler.transform(X_test))



mean_squared_error(y_predsvmReg,y_test)
adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predsvmReg)
svmReg = SVR(kernel='linear')

svmReg
svmReg.fit(X_train,y_train)

svmReg.score(X_train,y_train)
y_predsvmReg = svmReg.predict(scaler.transform(X_test))



mean_squared_error(y_predsvmReg,y_test)
adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predsvmReg)
svmReg = SVR(kernel='poly')

svmReg



svmReg.fit(X_train,y_train)

svmReg.score(X_train,y_train)



y_predsvmReg = svmReg.predict(scaler.transform(X_test))



mean_squared_error(y_predsvmReg,y_test)
adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predsvmReg)
gradBoostReg = GradientBoostingRegressor(loss='ls',n_estimators=500)

gradBoostReg
gradBoostReg.fit(X_train,y_train)

gradBoostReg.score(X_train,y_train)
y_predgradBoostReg = gradBoostReg.predict(scaler.transform(X_test))



mean_squared_error(y_predgradBoostReg,y_test)
adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predgradBoostReg)
XGBoostReg = xgb.XGBRegressor(objective="reg:linear",n_estimators=500)

XGBoostReg
XGBoostReg.fit(X_train,y_train)

XGBoostReg.score(X_train,y_train)
y_predXGBoostReg = XGBoostReg.predict(scaler.transform(X_test))



mean_squared_error(y_predXGBoostReg,y_test)
adjusted_r_square(X.shape[0],X.shape[1],y_test,y_predXGBoostReg)