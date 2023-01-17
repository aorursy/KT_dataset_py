#The following codes trains a regression model to predict the salaries of a set of position levels.

#The models tested are Linear Regression using Polynomial Fitting and Random Forests Regressor.

#An analysis is done to determine which model is a more accurate model for this application.



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#this line is to enable auto-prediction of commands

%config IPCompleter.greedy=True
pos_salary_df=pd.read_csv('../input/Position_Salaries.csv')
pos_salary_df
#identify and extract the independent variable

iv=pos_salary_df[['Level']]
#identify and extract the dependent variable (i.e. target)

dv=pos_salary_df[['Salary']]
#plot to visualise the raw data

plt.scatter(iv,dv)
#split data into training and test sets

from sklearn.model_selection import train_test_split

iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)
#see what is the identified independent variable training set

iv_train
#see what is the identified dependent variable training set

dv_train
#see what is the identified independent variable test set

iv_test
#see what is the identified dependent variable test set

dv_test
#Perform linear regression using Polynomial Fitting

#input the independent training data (iv_train) values into a polynomial of fifth degree

from sklearn.preprocessing import PolynomialFeatures



poly_transform=PolynomialFeatures(degree=5)

iv_poly=poly_transform.fit_transform(iv_train)
#display the independent variables transformed into the components of the polynomial

iv_poly
#do a polynomial fitting with the training data

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(iv_poly,dv_train)
#do linear regression prediction of test set dependent variable (actual values are in dv_test)

dv_poly_pred=lin_reg.predict(poly_transform.fit_transform(iv_test))
#Perform regression using Random Forests Regressor

from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor(n_estimators=10,random_state=0)

regressor.fit(iv_train,dv_train)
dv_rf_pred=regressor.predict(iv_test)
#check for goodness of fit for both Polynomial Fitting Linear Regression and Random Forests

from sklearn.metrics import mean_squared_error



rmse_poly = np.sqrt(mean_squared_error(dv_test,dv_poly_pred))

rmse_rf = np.sqrt(mean_squared_error(dv_test,dv_rf_pred))

print("root mean squared error using Polynomial Fitting:",rmse_poly)

print("root mean squared error using Random Forests Regression:",rmse_rf)



if (rmse_poly<rmse_rf):

   print("Linear regression using Polynomial Fitting is more accurate prediction model than Random Forests regression for this dataset.")

else:

   print("Random Forests regression is more accurate prediction model than Linear regression using Polynomial Fitting for this dataset.")
#Plot the predicted vs actual values over the full dataset, to visualise the "goodness of fit"

#from the plots, it seems to show that Linear Regression using Polynomial Fitting is more accurate compared to Random Forests Regressor

#The rmse values above also show the same conclusion.

#But if we do not set "n_estimators" and "random_state" parameters in RandomForestRegressor(), the resultant plots and rmse computations can change for every run,

#and sometimes rmse can even conclude that RandomForests is a more accurate model!



#plot the Random Forest predictions to visualise how well it fits the full dataset

plt.scatter(iv,dv)

plt.plot(iv,regressor.predict(iv),c='red',label="Random Forests predictions")



#plot polynomial fitting predictions to visualise how well it fits the full dataset

iv_poly_fullset=poly_transform.fit_transform(iv)

plt.plot(iv,lin_reg.predict(iv_poly_fullset),c='blue',label="Polynomial predictions")



plt.legend()