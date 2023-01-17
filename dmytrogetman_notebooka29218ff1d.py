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
table = pd.read_csv("../input/hazardous-asteroid-orbits/pha.csv")
table.head()
table.describe()
table.isnull().sum()
table.corr()
plt.figure(figsize = (15,10))
sns.heatmap(table.corr(),  annot = True)
X = table.loc[:,('a (AU)', 'e')].values
Y = table['Q (AU)']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)
LinearRegression = LinearRegression()
lin_reg = LinearRegression.fit(X_train, Y_train)
Y_pred = lin_reg.predict(X_test)
print('Predicted Q:', Y_pred)
accuracy = r2_score(Y_test,Y_pred)
print('Accuracy: ', accuracy)
MAE = mean_absolute_error(Y_test, Y_pred)
print('Absolute error:', MAE)
MRE = mean_squared_error(Y_test, Y_pred)
print('Squared error', MRE)
figure = plt.figure(figsize = (10,7))
plt.scatter(Y_test, Y_pred, alpha = 0.5, c = "b")
plt.xlabel('Actual Q (AU)')
plt.ylabel('Predicted Q (AU)')
figure = plt.figure(figsize = (10,7))
sns.distplot(Y_test - Y_pred, color = "blue")
Lasso = Lasso()
lasso = Lasso.fit(X_train, Y_train)
Y_pred_lasso = lasso.predict(X_test)
print('Predicted Q (AU) using lasso:', Y_pred_lasso)
accuracy_lasso = r2_score(Y_test, Y_pred_lasso)
print('Accuracy: ', accuracy_lasso)
MAE_lasso = mean_absolute_error(Y_test, Y_pred_lasso)
print('Absolute error:', MAE_lasso)
MRE_lasso = mean_squared_error(Y_test, Y_pred_lasso)
print('Squared error', MRE_lasso)
fig = plt.figure(figsize = (10,10))
plt.scatter(Y_test, Y_pred_lasso, alpha = 0.5, c = "b")
plt.xlabel('Actual Q (AU)')
plt.ylabel('Predicted Q (AU)')
fig = plt.figure(figsize=(10,10))
sns.distplot(Y_test-Y_pred_lasso,color="blue")
Ridge = Ridge()
ridge = Ridge.fit(X_train, Y_train)
Y_pred_ridge = ridge.predict(X_test)
print('Predicted Q (AU) using ridge:', Y_pred_ridge)
accuracy_ridge = r2_score(Y_test, Y_pred_ridge)
print('Accuracy: ', accuracy_ridge)
MAE_ridge = mean_absolute_error(Y_test, Y_pred_ridge)
print('Absolute error:', MAE_ridge)
MRE_ridge = mean_squared_error(Y_test, Y_pred_ridge)
print('Squared error', MRE_ridge)
fig = plt.figure(figsize = (10,10))
plt.scatter(Y_test, Y_pred_ridge, alpha = 0.5, c = "b")
plt.xlabel('Actual Q (AU)')
plt.ylabel('Predicted Q (AU)')
fig = plt.figure(figsize = (10,10))
sns.distplot(Y_test - Y_pred_ridge, color = "blue")
ElasticNet = ElasticNet()
elasticNet = ElasticNet.fit(X_train, Y_train)
Y_pred_elasticNet = elasticNet.predict(X_test)
print('Predicted Q (AU) using elasticNet:', Y_pred_elasticNet)
accuracy_elasticNet = r2_score(Y_test, Y_pred_elasticNet)
print('Accuracy: ', accuracy_elasticNet)
MAE_elasticNet = mean_absolute_error(Y_test, Y_pred_elasticNet)
print('Absolute error:', MAE_elasticNet)
MRE_elasticNet = mean_squared_error(Y_test, Y_pred_elasticNet)
print('Squared error', MRE_elasticNet)
fig = plt.figure(figsize = (10,10))
plt.scatter(Y_test, Y_pred_elasticNet, alpha = 0.5, c = "b")
plt.xlabel('Actual Q (AU)')
plt.ylabel('Predicted Q (AU)')
fig = plt.figure(figsize = (10,10))
sns.distplot(Y_test - Y_pred_elasticNet, color = "blue")
Ensamble = StackingRegressor([('LinearRegression', LinearRegression), ('Lasso', Lasso), ('Ridge', Ridge), ('ElasticNet', ElasticNet)])
ensamble = Ensamble.fit(X_train, Y_train)
Y_pred_ensamble = ensamble.predict(X_test)
print('Predicted Q (AU):',Y_pred_ensamble)
accuracy_ensamble = r2_score(Y_test,Y_pred_ensamble)
print('Accuracy: ', accuracy_ensamble)
MAE_ensamble = mean_absolute_error(Y_test,Y_pred_ensamble)
print('Absolute error:',MAE_ensamble)
MRE_ensamble = mean_squared_error(Y_test,Y_pred_ensamble)
print('Squared error',MRE_ensamble)
fig = plt.figure(figsize=(10,10))
plt.scatter(Y_test, Y_pred_ensamble, alpha = 0.5, c = "b")
plt.xlabel('Actual Q (AU)')
plt.ylabel('Predicted Q (AU)')
fig = plt.figure(figsize = (10,10))
sns.distplot(Y_test - Y_pred_ensamble, color = "blue")