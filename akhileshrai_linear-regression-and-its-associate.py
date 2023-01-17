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
import matplotlib.pyplot as plt

import seaborn as sns



dataset = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

X = dataset.iloc[:,:-1].values

Y = dataset.iloc[:,-1].values



correlation = dataset.corr()

fig = plt.subplots(figsize=(10,10))

sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Reds')



features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']

for feature in features:

    sns.set()

    sns.relplot(data = dataset,x = feature,y = Y, kind = 'line', height = 7, aspect = 1)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=4)

linear_regressor = LinearRegression()

linear_regressor.fit(x_train,y_train)

y_pred = linear_regressor.predict(x_test)



accuracy = linear_regressor.score(x_test,y_test)

print("Linear Accuracy: {}".format(accuracy))



rmse_linear = mean_squared_error(y_test,y_pred)

print("Linear RMSE: {} ".format(rmse_linear))
from sklearn.linear_model import Lasso



model2 = Lasso()

model2.fit(X=x_train, y=y_train)

y_pred2 = model2.predict(x_test)

rmse_lasso = mean_squared_error(y_pred2, y_test)

print("Lasso RMSE: {}".format(rmse_lasso))

accuracy_lasso = model2.score(x_test,y_test)

print("Accuracy Lasso: {}".format(accuracy_lasso))
from sklearn.linear_model import Ridge



model3 = Ridge(alpha = 1 , solver = "cholesky")

model3.fit(X=x_train, y=y_train)

y_pred3 = model3.predict(x_test)

rmse_ridge = mean_squared_error(y_pred3, y_test)

print("Ridge RMSE: {}".format(rmse_ridge))

accuracy_ridge = model3.score(x_test,y_test)

print("Accuracy Ridge: {}".format(accuracy_ridge))