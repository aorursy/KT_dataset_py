# Import necessary libraries 

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 



# Import the Height Weight Dataset 

data = pd.read_csv('HeightVsWeight.csv')

data.head(10)
#Store the data in the form of dependent and independent variables separately

X = data.iloc[:, 0:1].values 

y = data.iloc[:, 1].values 
#Split the Dataset into Training and Test Dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
#Fit the Simple Linear Regression Model

from sklearn.linear_model import LinearRegression

LinReg = LinearRegression()

LinReg.fit(X_train, y_train)
# Visualise the Linear Regression Result

plt.scatter(X_train, y_train, color = 'green') 

  

plt.plot(X_train, LinReg.predict(X_train), color = 'blue') 

plt.title('Linear Regression') 

plt.xlabel('Age') 

plt.ylabel('Height') 

  

plt.show() 
# Add the polynomial term to the equation/model

from sklearn.preprocessing import PolynomialFeatures 

  

polynom = PolynomialFeatures(degree = 2) 

X_polynom = polynom.fit_transform(X_train) 

  

X_polynom
#Fit the Polynomial Regression Model 

PolyReg = LinearRegression() 

PolyReg.fit(X_polynom, y_train) 
# Visualise the Polynomial Regression Results 

plt.scatter(X_train, y_train, color = 'green') 

  

plt.plot(X_train, PolyReg.predict(polynom.fit_transform(X_train)), color = 'blue') 

plt.title('Polynomial Regression') 

plt.xlabel('Age') 

plt.ylabel('Height') 

  

plt.show() 
#Predicted Height from test dataset w.r.t Simple Linear Regression

y_predict_slr = LinReg.predict(X_test)



#Model Evaluation using R-Square for Simple Linear Regression

from sklearn import metrics

r_square = metrics.r2_score(y_test, y_predict_slr)

print('R-Square Error associated with Simple Linear Regression:', r_square)
#Predicted Height from test dataset w.r.t Polynomial Regression

y_predict_pr = PolyReg.predict(polynom.fit_transform(X_test))



#Model Evaluation using R-Square for Polynomial Regression

from sklearn import metrics

r_square = metrics.r2_score(y_test, y_predict_pr)

print('R-Square Error associated with Polynomial Regression is:', r_square)
# Predicting Height based on Age using Linear Regression 

LinReg.predict([[53]]) 
# Predicting Height based on Age using Polynomial Regression 

PolyReg.predict(polynom.fit_transform([[53]])) 