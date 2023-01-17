# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

from sklearn.metrics import r2_score

import statsmodels.api as sm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/maaslar_yeni (1).csv")
data.info()
data.head() # We got the first 5 lines.
data.describe().T #Trampoz.
title_seniority_points=data.iloc[:,2:5].values 

salary=data.iloc[:,5:].values 

#We have values for forecasting and learning.
data.corr() #normalization process.
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(title_seniority_points,salary) 



#Linear OLS model

model=sm.OLS(lin_reg.predict(title_seniority_points),title_seniority_points)

model.fit().summary()
#Linear R2 Model

r2_score(salary,lin_reg.predict(title_seniority_points))
from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=4)

poly=poly_reg.fit_transform(title_seniority_points)



lin_reg2=LinearRegression()

lin_reg2.fit(poly,salary)



#Polynomial OLS model

model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(title_seniority_points)),title_seniority_points)

model2.fit().summary()

#data scaling

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

x_scaling=ss.fit_transform(title_seniority_points)

y_scaling=ss.fit_transform(salary)

from sklearn.svm import SVR

svr_reg = SVR( kernel = "rbf")

svr_reg.fit(x_scaling,y_scaling)



#SVR OLS model

model3=sm.OLS(svr_reg.predict(x_scaling),x_scaling)

model3.fit().summary()

#SVR R2 model

r2_score(y_scaling,svr_reg.predict(x_scaling))

#Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)

rf_reg.fit(title_seniority_points,salary)



#Random Forest Regression OLS model

model4=sm.OLS(rf_reg.predict(title_seniority_points),title_seniority_points)

model4.fit().summary()
#Random Forest Regression R2 model

r2_score(salary,rf_reg.predict(title_seniority_points))