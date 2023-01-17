import os

os.getcwd()
import numpy as np

import pandas as pd

df=pd.read_csv('../input/petrol-consumption/petrol_consumption.csv')

df.head()
df1=df

df.shape
q1=df1['Petrol_tax'].quantile(0.25)

q3=df1['Petrol_tax'].quantile(0.75)

iqr=q3-q1

print(iqr)

ll=q1-1.5*iqr

ul=q3+1.5*iqr

df1=df1[~((df['Petrol_tax']<ll) | (df1['Petrol_tax']>ul))]

df1.shape
q1=df1['Average_income'].quantile(0.25)

q3=df1['Average_income'].quantile(0.75)

iqr=q3-q1

print(iqr)

ll=q1-1.5*iqr

ul=q3+1.5*iqr

df1=df1[~((df1['Average_income']<ll) | (df1['Average_income']>ul))]

df1.shape
q1=df1['Paved_Highways'].quantile(0.25)

q3=df1['Paved_Highways'].quantile(0.75)

iqr=q3-q1

print(iqr)

ll=q1-1.5*iqr

ul=q3+1.5*iqr

df1=df1[~((df1['Paved_Highways']<ll) | (df1['Paved_Highways']>ul))]

df1.shape
q1=df1['Population_Driver_licence(%)'].quantile(0.25)

q3=df1['Population_Driver_licence(%)'].quantile(0.75)

iqr=q3-q1

print(iqr)

ll=q1-1.5*iqr

ul=q3+1.5*iqr

df1=df1[~((df1['Population_Driver_licence(%)']<ll) | (df1['Population_Driver_licence(%)']>ul))]

df1.shape
q1=df1['Petrol_Consumption'].quantile(0.25)

q3=df1['Petrol_Consumption'].quantile(0.75)

iqr=q3-q1

print(iqr)

ll=q1-1.5*iqr

ul=q3+1.5*iqr

df1=df1[~((df1['Petrol_Consumption']<ll) | (df1['Petrol_Consumption']>ul))]

df1.shape
df.corr()
df1.corr()
Y=df1.Petrol_Consumption

X=df1.drop(['Petrol_Consumption'],axis=1)
from sklearn.model_selection import train_test_split
Y=df1.Petrol_Consumption

X=df1.Petrol_tax

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state =0)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
import statsmodels.formula.api as smf

import statsmodels.api as sm

model=sm.OLS(Y_train,X_train).fit()

model.summary()
def mean_abs_per_err(pred,actual):

    per=(abs(actual-pred))/actual

    value=per*100

    return value
pred_y=model.predict(X_train)

from sklearn import metrics

MSE=metrics.mean_squared_error(pred_y,Y_train)

print('Mean Squared Error:',MSE)

RMSE=np.sqrt(MSE)

print('Root Mean Squared Error:',RMSE)

MAPE=(mean_abs_per_err(pred_y,Y_train).sum())/Y_train.size

print("Mean Absolute Percentage Error:", MAPE)
pred_y=model.predict(X_test)

from sklearn import metrics

MSE=metrics.mean_squared_error(pred_y,Y_test)

print('Mean Squared Error:',MSE)

RMSE=np.sqrt(MSE)

print('Root Mean Squared Error:',RMSE)

MAPE=(mean_abs_per_err(pred_y,Y_test).sum())/Y_test.size

print("Mean Absolute Percentage Error:", MAPE)
Y=df1.Petrol_Consumption

X=df1.drop(['Petrol_Consumption'],axis=1)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state =0)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model=sm.OLS(Y_train,X_train).fit()

model.summary()
pred_y=model.predict(X_train)

from sklearn import metrics

MSE=metrics.mean_squared_error(pred_y,Y_train)

print('Mean Squared Error:',MSE)

RMSE=np.sqrt(MSE)

print('Root Mean Squared Error:',RMSE)

MAPE=(mean_abs_per_err(pred_y,Y_train).sum())/Y_train.size

print("Mean Absolute Percentage Error:", MAPE)
pred_y=model.predict(X_test)

from sklearn import metrics

MSE=metrics.mean_squared_error(pred_y,Y_test)

print('Mean Squared Error:',MSE)

RMSE=np.sqrt(MSE)

print('Root Mean Squared Error:',RMSE)

MAPE=(mean_abs_per_err(pred_y,Y_test).sum())/Y_test.size

print("Mean Absolute Percentage Error:", MAPE)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, Y_train)
regressor.coef_