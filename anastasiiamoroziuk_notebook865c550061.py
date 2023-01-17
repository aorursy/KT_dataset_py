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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import StackingRegressor
data = pd.read_csv("../input/car-sales/Car_sales.csv")

data = data.drop('__year_resale_value',1)

data.info()
data.head()
data.describe()
data.isnull().sum()
data = data.dropna()

data.isnull().sum()
data.corr()
plt.figure(figsize=(15,10))

sns.heatmap(data.corr(), annot = True)
linreg=LinearRegression()

linreg


data.head(),data.shape
x=data.loc[:,('Horsepower','Power_perf_factor')].values

y=data['Price_in_thousands']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.5)

linearReg=linreg.fit(X_train,Y_train)

linearReg

Y_pred=linreg.predict(X_test)

print('Predicted car prices:',Y_pred)
acc = r2_score(Y_test,Y_pred)

print('Accuracy: ', acc)
mae = mean_absolute_error(Y_test,Y_pred)

print('Absolute error:',mae)

mre = mean_squared_error(Y_test,Y_pred)

print('Squared error',mre)
fig = plt.figure(figsize=(10,7))

plt.scatter(Y_test, Y_pred, alpha=0.5,c ="r")

plt.xlabel('Actual Price_in_thousands')

plt.ylabel('Predicted Price_in_thousands')
fig = plt.figure(figsize=(10,7))

sns.distplot(Y_test-Y_pred,color="red")
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.5)

lasso = Lasso()

reg_lasso = lasso.fit(X_train,Y_train)

Y_pred_lasso = reg_lasso.predict(X_test)

print('Predicted car prices using ridge:',Y_pred_lasso)
acc_lasso=r2_score(Y_test,Y_pred_lasso)

print('Accuracy: ', acc_lasso)
mae_lasso = mean_absolute_error(Y_test,Y_pred_lasso)

print('Absolute error:',mae_lasso)

mre_lasso = mean_squared_error(Y_test,Y_pred_lasso)

print('Squared error',mre_lasso)
fig = plt.figure(figsize=(10,7))

plt.scatter(Y_test, Y_pred_lasso, alpha=0.5,c="r")

plt.xlabel('Actual Price_in_thousands')

plt.ylabel('Predicted Price_in_thousands')

fig = plt.figure(figsize=(10,7))

sns.distplot(Y_test-Y_pred_lasso,color="red")
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.5)

ridge = Ridge()

reg_ridge = ridge.fit(X_train,Y_train)

Y_pred_ridge = reg_ridge.predict(X_test)

print('Predicted car prices using ridge:',Y_pred_ridge)
acc_ridge = r2_score(Y_test,Y_pred_ridge)

print('Accuracy: ', acc_ridge)
mae_ridge = mean_absolute_error(Y_test,Y_pred_ridge)

print('Absolute error:',mae_ridge)

mre_ridge = mean_squared_error(Y_test,Y_pred_ridge)

print('Squared error',mre_ridge)
fig = plt.figure(figsize=(10,7))

plt.scatter(Y_test, Y_pred_ridge, alpha=0.5,c="r")

plt.xlabel('Actual Price_in_thousands')

plt.ylabel('Predicted Price_in_thousands')
fig = plt.figure(figsize=(10,7))

sns.distplot(Y_test-Y_pred_ridge,color="red")
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.5)

elasticNet = ElasticNet()

reg_elasticNet = elasticNet.fit(X_train,Y_train)

Y_pred_elasticNet = reg_elasticNet.predict(X_test)

print('Predicted car prices using ridge:',Y_pred_elasticNet)
acc_elasticNet = r2_score(Y_test,Y_pred_elasticNet)

print('Accuracy: ', acc_elasticNet)
mae_elasticNet = mean_absolute_error(Y_test,Y_pred_elasticNet)

print('Absolute error:',mae_elasticNet)

mre_elasticNet = mean_squared_error(Y_test,Y_pred_elasticNet)

print('Squared error',mre_elasticNet)
fig = plt.figure(figsize=(10,7))

plt.scatter(Y_test, Y_pred_elasticNet, alpha=0.5, c="r")

plt.xlabel('Actual Price_in_thousands')

plt.ylabel('Predicted Price_in_thousands')
fig = plt.figure(figsize=(10,7))

sns.distplot(Y_test-Y_pred_elasticNet,color="red")
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.5)

ensamble=StackingRegressor([('LinearRegression',linearReg),('Lasso',lasso),('Ridge',ridge),('ElasticNet',elasticNet)])

reg_ensamble=ensamble.fit(X_train, Y_train)

Y_pred_ensamble = ensamble.predict(X_test)

print('Predicted car prices using ridge:',Y_pred_ensamble)
acc_ensamble =r2_score(Y_test,Y_pred_ensamble)

print('Accuracy: ', acc_ensamble)
mae_ensamble = mean_absolute_error(Y_test,Y_pred_ensamble)

print('Absolute error:',mae_ensamble)

mre_ensamble = mean_squared_error(Y_test,Y_pred_ensamble)

print('Squared error',mre_ensamble)
fig = plt.figure(figsize=(10,7))

plt.scatter(Y_test, Y_pred_ensamble, alpha=0.5, c="r")

plt.xlabel('Actual Price_in_thousands')

plt.ylabel('Predicted Price_in_thousands')
fig = plt.figure(figsize=(10,7))

sns.distplot(Y_test-Y_pred, color="red")