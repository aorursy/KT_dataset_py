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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

sns.set_style('whitegrid')

%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col=0)

pd.set_option('display.max_columns', None)
train.head()
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col=0)
test.head()
train.info()
test.info()
plt.figure(figsize=(15,5))

sns.distplot(train['SalePrice'])
sns.barplot(x='MSSubClass',y='SalePrice',data=train,palette='BrBG')
sns.jointplot(x='LotArea',y='SalePrice',data=train,kind='hex')
sns.barplot(x='LandContour',y='SalePrice',data=train,palette='BrBG')
sns.barplot(x='PoolArea',y='SalePrice',data=train,palette='BrBG')
sns.barplot(x='BedroomAbvGr',y='SalePrice',data=train,palette='BrBG')
sns.barplot(x='KitchenAbvGr',y='SalePrice',data=train,palette='BrBG')
sns.barplot(x='FullBath',y = 'SalePrice',data=train,palette='BrBG')
sns.jointplot(x='GarageArea',y='SalePrice',data=train,kind='hex')
train = train.select_dtypes(include=['int'])
train.head()
train.columns
plt.figure(figsize=(20,10))

sns.heatmap(train.corr(),annot=False,cmap='Blues')
train.corrwith(train['SalePrice'])
train.isna().sum()
train.isnull().sum()
from sklearn.model_selection import train_test_split





X = train.drop('SalePrice',axis=1)



y = train['SalePrice']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# linear regression



from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train,y_train)

lr_pred = lr.predict(X_test)
print("Intercept: {}".format(lr.intercept_))

print("Coeffs: {}".format(lr.coef_))
plt.figure(figsize=(15,5))

plt.scatter(y_test,lr_pred)
plt.figure(figsize=(15,5))

sns.distplot(y_test-lr_pred)
x_ax = range(len(X_test))

plt.figure(figsize=(15,5))

plt.scatter(x_ax, y_test, s=5, color="blue", label="original")

plt.plot(x_ax, lr_pred, lw=0.8, color="red", label="predicted")

plt.legend()

plt.show()
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



mae = mean_absolute_error(y_test,lr_pred)

mse = mean_squared_error(y_test,lr_pred)

rmse = np.sqrt(mean_squared_error(y_test,lr_pred))
print("Mean Absloute Error: {}".format(mae))

print("Mean Squared Error: {}".format(mse))

print("RMSE: {}".format(rmse))
from sklearn.tree import DecisionTreeRegressor



dtree = DecisionTreeRegressor(random_state=100)

dtree.fit(X_train,y_train)

dtree_pred=dtree.predict(X_test)
plt.figure(figsize=(15,5))

plt.scatter(y_test,dtree_pred)
plt.figure(figsize=(15,5))

sns.distplot(y_test-dtree_pred)
x_ax = range(len(X_test))

plt.figure(figsize=(15,5))

plt.scatter(x_ax, y_test, s=5, color="blue", label="original")

plt.plot(x_ax, dtree_pred, lw=0.8, color="red", label="predicted")

plt.legend()

plt.show()
mae = mean_absolute_error(y_test,dtree_pred)

mse = mean_squared_error(y_test,dtree_pred)

rmse = np.sqrt(mean_squared_error(y_test,dtree_pred))



print("Mean Absloute Error: {}".format(mae))

print("Mean Squared Error: {}".format(mse))

print("RMSE: {}".format(rmse))
from sklearn.ensemble import RandomForestRegressor



rfr = RandomForestRegressor(n_estimators=110)

rfr.fit(X_train,y_train)

rfr_pred = rfr.predict(X_test)
plt.figure(figsize=(15,5))

plt.scatter(y_test,rfr_pred)
plt.figure(figsize=(15,5))

sns.distplot(y_test-rfr_pred)
x_ax = range(len(X_test))

plt.figure(figsize=(15,5))

plt.scatter(x_ax, y_test, s=5, color="blue", label="original")

plt.plot(x_ax, rfr_pred, lw=0.8, color="red", label="predicted")

plt.legend()

plt.show()
mae = mean_absolute_error(y_test,rfr_pred)

mse = mean_squared_error(y_test,rfr_pred)

rmse = np.sqrt(mean_squared_error(y_test,rfr_pred))



print("Mean Absloute Error: {}".format(mae))

print("Mean Squared Error: {}".format(mse))

print("RMSE: {}".format(rmse))
from sklearn.svm import SVR



sv = SVR(kernel = 'rbf')

#sv.fit(X_train,y_train)
from sklearn.model_selection import GridSearchCV



grid_param = {'C': [0.1,0.01,1,10,100,1000,10000],'gamma':[1,0.1,0.01,0.001,0.0001,0.00001,0.000001]}

grid = GridSearchCV(SVR(),grid_param,verbose = 3)
grid.fit(X_train,y_train)
grid.best_estimator_
grid.best_params_
grid_pred = grid.predict(X_test)
plt.figure(figsize=(15,5))

plt.scatter(y_test,grid_pred)
plt.figure(figsize=(15,5))

sns.distplot(y_test-grid_pred)
x_ax = range(len(X_test))

plt.figure(figsize=(15,5))

plt.scatter(x_ax, y_test, s=5, color="blue", label="original")

plt.plot(x_ax, grid_pred, lw=0.8, color="red", label="predicted")

plt.legend()

plt.show()
mae = mean_absolute_error(y_test,grid_pred)

mse = mean_squared_error(y_test,grid_pred)

rmse = np.sqrt(mean_squared_error(y_test,grid_pred))



print("Mean Absloute Error: {}".format(mae))

print("Mean Squared Error: {}".format(mse))

print("RMSE: {}".format(rmse))
from sklearn.linear_model import Ridge



ridge = Ridge()
ridge_params = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

ridge_grid = GridSearchCV(ridge,ridge_params,scoring='r2',cv=5,verbose=3)

ridge_grid.fit(X_train,y_train)



print(ridge_grid.best_params_)

print(ridge_grid.best_score_)
ridge_predict = ridge_grid.predict(X_test)
plt.figure(figsize=(15,5))

plt.scatter(y_test,ridge_predict)
plt.figure(figsize=(15,5))

sns.distplot(y_test-ridge_predict)
x_ax = range(len(X_test))

plt.figure(figsize=(15,5))

plt.scatter(x_ax, y_test, s=5, color="blue", label="original")

plt.plot(x_ax, ridge_predict, lw=0.8, color="red", label="predicted")

plt.legend()

plt.show()
mae = mean_absolute_error(y_test,ridge_predict)

mse = mean_squared_error(y_test,ridge_predict)

rmse = np.sqrt(mean_squared_error(y_test,ridge_predict))



print("Mean Absloute Error: {}".format(mae))

print("Mean Squared Error: {}".format(mse))

print("RMSE: {}".format(rmse))

from sklearn.linear_model import Lasso



lasso = Lasso()

#lasso.fit(X_train,y_train)
lasso_params = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

grid_lasso = GridSearchCV(lasso,lasso_params,scoring='r2',cv=5,verbose=3)

grid_lasso.fit(X_train,y_train)



print(grid_lasso.best_params_)

print(grid_lasso.best_score_)
lasso_predict = grid_lasso.predict(X_test)
plt.figure(figsize=(15,5))

plt.scatter(y_test,lasso_predict)
plt.figure(figsize=(15,5))

sns.distplot(y_test-lasso_predict)
x_ax = range(len(X_test))

plt.figure(figsize=(15,5))

plt.scatter(x_ax, y_test, s=5, color="blue", label="original")

plt.plot(x_ax, lasso_predict, lw=0.8, color="red", label="predicted")

plt.legend()

plt.show()
mae = mean_absolute_error(y_test,lasso_predict)

mse = mean_squared_error(y_test,lasso_predict)

rmse = np.sqrt(mean_squared_error(y_test,lasso_predict))



print("Mean Absloute Error: {}".format(mae))

print("Mean Squared Error: {}".format(mse))

print("RMSE: {}".format(rmse))

from sklearn.linear_model import ElasticNet





elastic=ElasticNet()

elastic_params = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

grid_elastic = GridSearchCV(elastic,elastic_params,scoring='r2',cv=5,verbose=3)

grid_elastic.fit(X_train,y_train)







elastic_predict = grid_elastic.predict(X_test)
plt.figure(figsize=(15,5))

plt.scatter(y_test,elastic_predict)
plt.figure(figsize=(15,5))

sns.distplot(y_test-elastic_predict)
x_ax = range(len(X_test))

plt.figure(figsize=(15,5))

plt.scatter(x_ax, y_test, s=5, color="blue", label="original")

#plt.plot(x_ax, y_test, color="blue", label="original")

plt.plot(x_ax, elastic_predict, lw=0.8, color="red", label="predicted")

plt.legend()

plt.show()
mae = mean_absolute_error(y_test,elastic_predict)

mse = mean_squared_error(y_test,elastic_predict)

rmse = np.sqrt(mean_squared_error(y_test,elastic_predict))



print("Mean Absloute Error: {}".format(mae))

print("Mean Squared Error: {}".format(mse))

print("RMSE: {}".format(rmse))