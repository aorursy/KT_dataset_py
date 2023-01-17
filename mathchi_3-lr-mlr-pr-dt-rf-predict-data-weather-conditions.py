# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

import operator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Sum_Weather = pd.read_csv("../input/weatherww2/Summary of Weather.csv")
Sum_Weather.head(2)
Sum_Weather.info()
Sum_Weather.describe().T
# Extract 3 columns 'MaxTemp','MinTemp', 'MeanTemp' for pure and better showing



Sum_Weather_df = Sum_Weather[['MaxTemp','MinTemp', 'Date']]

#Sum_Weather_df
Sum_Weather_df = Sum_Weather_df[:][:500]      # lets take limit for speed regression calculating

Sum_Weather_df.head(2)
pd.to_datetime(Sum_Weather_df['Date'])
# See picture with scatter or plot method



#sns.pairplot(Sum_Weather_df, kind="reg")



plt.figure(figsize=(22,10))

plt.plot(Sum_Weather_df.Date, Sum_Weather_df.MaxTemp, Sum_Weather_df.MinTemp,)

plt.title("Max and Min Temperature of Dates")

plt.xlabel("Date")

plt.ylabel("Max and Min Temperature")

plt.legend()

plt.show()



# see how many null values we have



Sum_Weather_df.isnull().sum()
# Features chose



y = np.array(Sum_Weather_df['MaxTemp']).reshape(-1, 1)

X = np.array(Sum_Weather_df['MinTemp']).reshape(-1, 1)
# Split data as %20 is test and %80 is train set



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
from sklearn.linear_model import LinearRegression



lin_df = LinearRegression()  

lin_df.fit(X_train, y_train)
y_pred = lin_df.predict(X_test)                                     # Predict Linear Model

accuracy_score = lin_df.score(X_test, y_test)                       # Accuracy score

print("Linear Regression Model Accuracy Score: " + "{:.1%}".format(accuracy_score))
from sklearn.metrics import mean_squared_error,r2_score



print("R2 Score: " +"{:.3}".format(r2_score(y_test, y_pred)));
# Finally draw figure of Linear Regression Model



plt.scatter(X_test, y_test, color='r')

plt.plot(X_test, y_pred, color='g')

plt.show()
mlin_df = LinearRegression()

mlin_df = mlin_df.fit(X_train, y_train)

mlin_df.intercept_       # constant b0

mlin_df.coef_            # variable coefficient
y_pred = mlin_df.predict(X_train)                                      # predict Multi linear Reg model

rmse = np.sqrt(mean_squared_error(y_train, mlin_df.predict(X_train)))

print("RMSE Score for Test set: " +"{:.2}".format(rmse))

print("R2 Score for Test set: " +"{:.3}".format(r2_score(y_train, y_pred)));      # this is test error score
# cross validation method is giving better and clear result



cross_val_score(mlin_df, X, y, cv=10, scoring = 'r2').mean()
mlin_df.score(X_train, y_train)      # r2 value
# Finally draw figure of Multiple Linear Regression Model



plt.scatter(X_train, y_train, s=100)



# sort the values of x before line plot

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(X_train,y_pred), key=sort_axis)

X_test, y_pred = zip(*sorted_zip)

plt.plot(X_train, y_train, color='r')

plt.show()
from sklearn.preprocessing import PolynomialFeatures



poly_df = PolynomialFeatures(degree = 5)

transform_poly = poly_df.fit_transform(X_train)



linreg2 = LinearRegression()

linreg2.fit(transform_poly,y_train)



polynomial_predict = linreg2.predict(transform_poly)
rmse = np.sqrt(mean_squared_error(y_train,polynomial_predict))

r2 = r2_score(y_train,polynomial_predict)

print("RMSE Score for Test set: " +"{:.2}".format(rmse))

print("R2 Score for Test set: " +"{:.2}".format(r2))
plt.scatter(X_train, y_train, s=50)

# sort the values of x before line plot

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(X_train,polynomial_predict), key=sort_axis)

X_train, polynomial_predict = zip(*sorted_zip)

plt.plot(X_train, polynomial_predict, color='m')

plt.show()
from sklearn.tree import DecisionTreeRegressor



dt_reg = DecisionTreeRegressor()          # create  DecisionTreeReg with sklearn

dt_reg.fit(X_train,y_train)
dt_predict = dt_reg.predict(X_train)

#dt_predict.mean()
plt.scatter(X_train,y_train, color="red")                           # scatter draw

X_grid = np.arange(min(np.array(X_train)),max(np.array(X_train)), 0.01)  

X_grid = X_grid.reshape((len(X_grid), 1))

plt.plot(X_grid,dt_reg.predict(X_grid),color="g")                 # line draw

plt.xlabel("Temperature")

plt.ylabel("Salinity")

plt.title("Decision Tree Model")

plt.show()
rmse = np.sqrt(mean_squared_error(y_train,dt_predict))

r2 = r2_score(y_train,dt_predict)

print("RMSE Score for Test set: " +"{:.2}".format(rmse))

print("R2 Score for Test set: " +"{:.2}".format(r2))
from sklearn.ensemble import RandomForestRegressor



rf_reg = RandomForestRegressor(n_estimators=5, random_state=0)

rf_reg.fit(X_train,y_train)

rf_predict = rf_reg.predict(X_train)

#rf_predict.mean()
plt.scatter(X_train,y_train, color="red")                           # scatter draw

X_grid = np.arange(min(np.array(X_train)),max(np.array(X_train)), 0.01)  

X_grid = X_grid.reshape((len(X_grid), 1))

plt.plot(X_grid,rf_reg.predict(X_grid),color="b")                 # line draw

plt.xlabel("Temperature")

plt.ylabel("Salinity")

plt.title("Random Forest Model")

plt.show()
rmse = np.sqrt(mean_squared_error(y_train,rf_predict))

r2 = r2_score(y_train,rf_predict)

print("RMSE Score for Test set: " +"{:.2}".format(rmse))

print("R2 Score for Test set: " +"{:.2}".format(r2))