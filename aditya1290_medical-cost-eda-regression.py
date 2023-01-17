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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv('../input/insurance.csv')
dataset.shape
dataset.columns
dataset.info()
dataset.describe()
dataset.describe(include = ['O'])
dataset['region'].unique()
dataset['sex'].unique()
dataset['smoker'].unique()
#you can do it by mapping

dataset['sex'] = dataset['sex'].map({'male':1,'female':0})



#you can do by looping

dataset.smoker = [1 if each=='yes' else 0 for each in dataset.smoker]
dataset.head(10)
dataset.describe()
dataset_normal = dataset.copy()

dataset_normal['bmi'] = (dataset_normal['bmi'] - np.min(dataset_normal['bmi']))/(np.max(dataset_normal['bmi']) - np.min(dataset_normal['bmi']))

dataset_normal['charges'] = (dataset_normal['charges'] - np.min(dataset_normal['charges']))/(np.max(dataset_normal['charges']) - np.min(dataset_normal['charges']))
dataset_normal.head(10)
dummy = pd.get_dummies(dataset_normal['region'],drop_first = True)
dataset_normal = pd.concat([dataset_normal,dummy],axis=1)
dataset_normal.drop('region',axis=1,inplace = True)
dataset_normal.head(15)
x = dataset_normal.copy()

x.drop('charges',axis=1,inplace=True)
y = dataset_normal.iloc[:,5:6]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
from sklearn.linear_model import LinearRegression

lr_mlr = LinearRegression()

lr_mlr.fit(x_train,y_train)
y_pred_lr = lr_mlr.predict(x_test)

plt.figure(figsize=(10,7))

c = [i for i in range(402)]

plt.plot(c,y_test-y_pred_lr,'-')

plt.show()
from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test,y_pred_lr)

r2 = r2_score(y_test,y_pred_lr)

print(mse)

print(r2)
import statsmodels.api as sm

x_train_sm = sm.add_constant(x_train)

lr_sm_1 = sm.OLS(y_train,x_train_sm).fit()

print(lr_sm_1.summary())
x_train_sm.drop('sex',axis=1,inplace=True)

import statsmodels.api as sm

lr_sm_2 = sm.OLS(y_train,x_train_sm).fit()

print(lr_sm_2.summary())
x_train_sm.drop('northwest',axis=1,inplace=True)

import statsmodels.api as sm

lr_sm_3 = sm.OLS(y_train,x_train_sm).fit()

print(lr_sm_3.summary())
x_train_sm.drop(['southeast','southwest'],axis=1,inplace=True)

import statsmodels.api as sm

lr_sm_4 = sm.OLS(y_train,x_train_sm).fit()

print(lr_sm_4.summary())
from sklearn.preprocessing import PolynomialFeatures

lr_pr = PolynomialFeatures(degree=3)

x_train_poly = lr_pr.fit_transform(x_train)

x_test_poly = lr_pr.fit_transform(x_test)

lr_pr1 = LinearRegression()

lr_pr1.fit(x_train_poly,y_train)
y_pred_poly = lr_pr1.predict(x_test_poly)
c = [i for i in range(402)]

plt.figure(figsize=(10,7))

plt.plot(c,y_test-y_pred_poly,'-')

plt.show()
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test,y_pred_poly)

r2 = r2_score(y_test,y_pred_poly)

print(mse)

print(r2)
from sklearn.svm import SVR

lr_svr = SVR(kernel='linear')

lr_svr.fit(x_train,y_train)
y_pred_svr = lr_svr.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test,y_pred_svr)

r2 = r2_score(y_test,y_pred_svr)

print(mse)

print(r2)
from sklearn.tree import DecisionTreeRegressor

lr_dtr = DecisionTreeRegressor(random_state=0)

lr_dtr.fit(x_train,y_train)
y_pred_dtr = lr_dtr.predict(x_test)

plt.figure(figsize=(10,7))

plt.plot(c,y_test,'-')

plt.plot(c,y_pred_dtr,'-o')

plt.show()
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test,y_pred_dtr)

r2 = r2_score(y_test,y_pred_dtr)

print(mse)

print(r2)
from sklearn.ensemble import RandomForestRegressor

lr_rfr = RandomForestRegressor(n_estimators=100,max_depth=7,random_state=100)

lr_rfr.fit(x_train,y_train)
y_pred_rfr = lr_rfr.predict(x_test)

plt.figure(figsize=(10,7))

plt.plot(c,y_test,'-')

plt.plot(c,y_pred_rfr,'-o')

plt.show()
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test,y_pred_rfr)

r2 = r2_score(y_test,y_pred_rfr)

print(mse)

print(r2)