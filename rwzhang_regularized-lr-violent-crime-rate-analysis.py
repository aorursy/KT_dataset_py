# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np
import operator
from sklearn import linear_model,metrics
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/communities.data', header=None)
data=data.drop([3], axis=1)
num_row, num_col = data.shape
selected_column = []

for j in range(num_col-1):
    valcount = data.iloc[:,j].value_counts()
    if '?' not in valcount:
        selected_column.append(j)
    elif valcount['?'] < 0.01 * num_row:
        valmean = pd.to_numeric(data.iloc[:,j], errors='coerce').mean()
        for i in range(num_row):
            if data.iloc[i,j] == '?':
                data.iloc[i,j] = valmean
        data.iloc[:,j] = pd.to_numeric(data.iloc[:,j])
        selected_column.append(j)

np.random.seed(2018)
train = np.random.choice([True, False], num_row, replace=True, p=[0.9,0.1])
x_train = data.iloc[train,selected_column].as_matrix()
y_train = data.iloc[train,-1].as_matrix()
x_test = data.iloc[~train,selected_column].as_matrix()
y_test = data.iloc[~train,-1].as_matrix()
x=np.r_[x_train,x_test]
y=np.r_[y_train,y_test]
#build linear model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_predict=regr.predict(x_test)
mse=metrics.mean_squared_error(y_test, y_predict)
dictionary_linear={}
for a in [x for x in range(len(regr.coef_)) if x != 3]:
    dictionary_linear[a]=abs(regr.coef_[a])
sorted_linear = sorted(dictionary_linear.items(), key=operator.itemgetter(1),reverse=True)
print('\t','linear regression model:')
print('MSE on test set:', mse)
print('important attributes:',sorted_linear[:2])
def my_linspace(min_value, max_value, steps):
    diff = max_value - min_value
    return np.linspace(min_value - 0.1 * diff, max_value + 0.1 * diff, steps)

steps = 200
x0 = my_linspace(min(x[:,66]), max(x[:,66]), steps)
x1 = my_linspace(min(x[:,43]), max(x[:,43]), steps)
xx0, xx1 = np.meshgrid(x0, x1)
mesh_data = np.c_[xx0.ravel(), xx1.ravel()]
regr1 = linear_model.LinearRegression()
regr1.fit(x_train[:,[66,43]], y_train)
mesh_y = regr1.predict(mesh_data).reshape(steps,steps)
plt.contourf(xx0, xx1, mesh_y, 20, cmap=plt.cm.Greys, alpha=0.5)
plt.show()
alphas = [ 0.001,0.01, 0.1, 1]
for i in range(4):
    #build lasso here   
    lasso_cofficients = []
    lasso = linear_model.Lasso(alpha = alphas[i])
    lasso.fit(x_train, y_train)
    lasso_cofficients.append(lasso.coef_)
    y_predict = lasso.predict(x_test)
    mse=metrics.mean_squared_error(y_test, y_predict)
    cplxy=0
    (num_coef,)=lasso.coef_.shape
    for a in range(num_coef):
        cplxy=cplxy+alphas[i]*abs(lasso.coef_[a,])
    dictionary_lasso={}
    for l in  range(len(regr.coef_)) :
        dictionary_lasso[l]=abs(regr.coef_[l])
    sorted_lasso = sorted(dictionary_lasso.items(), key=operator.itemgetter(1),reverse=True)
    print('\t','lasso model: ','alpha=',alphas[i])
    print('MSE on test set:', mse)
    print('MSE on test set:', mse)
    print('model complexity:', cplxy)
    print('regularized cost:', mse + cplxy)
    print('important attributes:',sorted_lasso[:2])
steps = 200
x0 = my_linspace(min(x[:,66]), max(x[:,66]), steps)
x1 = my_linspace(min(x[:,43]), max(x[:,43]), steps)
xx0, xx1 = np.meshgrid(x0, x1)
mesh_data = np.c_[xx0.ravel(), xx1.ravel()]
print(mesh_data.shape)
lasso1 = linear_model.Lasso(alpha = 0.001)
lasso1.fit(x[:,[66,43]], y)
mesh_y = lasso1.predict(mesh_data).reshape(steps,steps)
print(mesh_y)
axes = plt.gca()
plt.contourf(xx0, xx1, mesh_y, 20, cmap=plt.cm.Greys, alpha=0.5)
plt.show()
for i in range(4):
    #build ridge here
    ridge_cofficients = []
    ridge = linear_model.Ridge(alpha = alphas[i])
    ridge.fit(x_train, y_train)
    ridge_cofficients.append(ridge.coef_)
    y_predict = ridge.predict(x_test)
    mse=metrics.mean_squared_error(y_test, y_predict)
    cplxy=0
    (num_coef,)=ridge.coef_.shape
    for a in range(num_coef):
        cplxy=cplxy+alphas[i]*np.square((ridge.coef_[a,]))
    dictionary_ridge={}
    for l in [x for x in range(len(ridge.coef_)) if x != 3]:
        dictionary_ridge[l]=abs(ridge.coef_[l])
    sorted_ridge = sorted(dictionary_ridge.items(), key=operator.itemgetter(1),reverse=True)
    print('\t','ridge model: ','alpha=',alphas[i])
    print('MSE on test set:', mse)
    print('model complexity:', cplxy)
    print('regularized cost:', mse + cplxy)
    print('important attributes:',sorted_ridge[:2])
steps = 200
x0 = my_linspace(min(x[:,66]), max(x[:,66]), steps)
x1 = my_linspace(min(x[:,43]), max(x[:,43]), steps)
xx0, xx1 = np.meshgrid(x0, x1)
mesh_data = np.c_[xx0.ravel(), xx1.ravel()]
print(mesh_data.shape)
ridge1 = linear_model.Ridge(alpha = 0.001)
ridge1.fit(x[:,[66,43]], y)
mesh_y = ridge1.predict(mesh_data).reshape(steps,steps)
plt.contourf(xx0, xx1, mesh_y, 20, cmap=plt.cm.Greys, alpha=0.5)
plt.show()