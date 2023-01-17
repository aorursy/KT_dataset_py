# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/salary/Salary.csv')

xi = df.iloc[:,0].values

yi = df.iloc[:,1].values
xi= xi.reshape(1,-1).transpose()

yi= yi.reshape(1,-1).transpose()
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(xi,yi,test_size = 0.2,random_state = 0)
#LinearRegression



from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



Regr = LinearRegression()

LrModel = Regr.fit(x_train,y_train)



y_pred = LrModel.predict(x_test)



comparison = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})

print(comparison)



plt.scatter(x_test,y_test)

plt.plot(x_test,y_pred,color='red')

plt.show()
#SVM

from sklearn.svm import SVR

import matplotlib.pyplot as plt





sv_reg =  SVR(kernel = 'rbf', C=10000, gamma=0.1, epsilon=.1)

sv_model = sv_reg.fit(x_train,y_train)



sv_pred = sv_model.predict(x_test)



sv_comparison = pd.DataFrame({'Actual':y_test.flatten(),'SVR':sv_pred.flatten()})

sv_comparison
plt.scatter(x_test,y_test)

plt.plot(x_test,sv_pred,color='red')

plt.show()
#Polynomial Regression



from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



poly_reg = PolynomialFeatures(degree = 2)

x_poly = poly_reg.fit_transform(x_train)

poly_reg.fit(x_poly,y_train)



lr = LinearRegression()

model = lr.fit(x_poly,y_train)



y_poly = model.predict(poly_reg.fit_transform(x_test))



poly_Compare = pd.DataFrame({'Actual':y_test.flatten(),'Linear':y_pred.flatten(),'SVR':sv_pred.flatten(),'Poly':y_poly.flatten()})



print(poly_Compare)



plt.scatter(x_test,y_test)

plt.plot(x_test,y_poly, color='red')

plt.show()
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt



dt_reg = DecisionTreeRegressor(random_state = 0)

dt_model = dt_reg.fit(x_train,y_train)



y_dt = dt_model.predict(x_test)

dt_Compare = pd.DataFrame({'Actual':y_test.flatten(),'Linear':y_pred.flatten(),

                             'SVR':sv_pred.flatten(),'Poly':y_poly.flatten(),'DecisionTree':y_dt.flatten()})



#xgrid = np.arange(min(x_test),max(x_test),0.01)

#xgrid = xgrid.reshape((len(xgrid),1))

plt.scatter(x_test,y_test)

plt.plot(x_test,y_dt)

plt.show()
#Randomforest

from sklearn.ensemble import RandomForestRegressor



rf_regr =  RandomForestRegressor(n_estimators = 10, random_state = 0)

rf_model = rf_regr.fit(x_train,y_train)



y_rf = rf_model.predict(x_test)

y_rf = y_rf.reshape(1,-1).transpose()



y_dt = dt_model.predict(x_test)

dt_Compare = pd.DataFrame({'Actual':y_test.flatten(),'Linear':y_pred.flatten(),

                             'SVR':sv_pred.flatten(),'Poly':y_poly.flatten(),

                           'DecisionTree':y_dt.flatten(),'randomforest':y_rf.flatten()})



plt.scatter(x_test,y_test)

plt.plot(x_test,y_rf, color='green')

plt.show()

dt_Compare