# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



#Importing Datasets

ds = pd.read_csv("../input/bottle.csv")





ds = ds[:][:500]

ds.head()
#Getting rid of Nan data

temp = ds[ds['Salnty'].isnull()].index

ds = ds.drop(temp)



temp = ds[ds['T_degC'].isnull()].index

ds = ds.drop(temp)



#Usable Variable Data

x = ds['Salnty']



x = np.array(x)

x = np.reshape(x,(len(x),1))



y = ds['T_degC']



x

y
#Splitting Data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,

                                                 random_state = 20)

#Initial Visualisation

plt.scatter(x,y)
#Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x_train,y_train)



#Visaulising Linear regression Results

plt.scatter(x_test,y_test,color = 'red')

plt.plot(x_test,lin_reg.predict(x_test),color = 'blue')
#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

poly_feat = PolynomialFeatures(degree = 2)

x_poly = poly_feat.fit_transform(x_train)

poly_reg = LinearRegression()

poly_reg.fit(x_poly,y_train)



#Ensure smooth curve

x_grid = np.arange(min(x_train),max(x_train),0.01)

x_grid = x_grid.reshape(len(x_grid),1)



#Visualising Polynominal Results

plt.scatter(x_test,y_test,color = 'red')

plt.plot(x_grid,poly_reg.predict(poly_feat.fit_transform(x_grid)),color = 'blue')

#Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(min_samples_split =50, random_state = 0)

dt_reg.fit(x_train,y_train)



#Visualising Decision Tree Regression

plt.scatter(x_test, y_test, color = 'red')

plt.plot(x_grid, dt_reg.predict(x_grid), color = 'blue')



#Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 1,min_samples_split=20, random_state = 0)

rf_reg.fit(x_train,y_train)



#Visualising Random Forest Regression

plt.scatter(x_test, y_test, color = 'red')

plt.plot(x_grid, rf_reg.predict(x_grid), color = 'blue')
