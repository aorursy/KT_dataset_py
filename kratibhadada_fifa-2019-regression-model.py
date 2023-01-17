import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns
# reading dataset 

data=pd.read_csv("../input/data.csv")
# displaying first 5 rows

data.head()
data.shape #(no. of rows, no. of columns)
data.describe()
# finding any null values in data

data.isnull().any()
# x = Age(independent variable)

x=data.iloc[:,3] 
x.head()
x.isnull().any()
# y = Potential(dependent variable)

y=data.iloc[:,8]
y.head()
y.isnull().any()
plt.bar(data["Age"],data["Potential"])

plt.xlabel("Age of Player")

plt.show()
# splitting data into train and tet set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
# making object regressor of class LinearRegression

regressor=LinearRegression()
type(x_train)

type(y_train)
x_train=np.array(x_train)

y_train=np.array(y_train)
type(x_train)

type(y_train)
x_train=x_train.reshape(-1,1)

y_train=y_train.reshape(-1,1)

# fitting training set into object regressor

regressor.fit(x_train,y_train)
x_test=np.array(x_test)
x_test=x_test.reshape(-1,1)
# Predicting y from test set

y_pred= regressor.predict(x_test)
# Visualising training dataset

plt.scatter(x_train,y_train,color="red")

plt.xlabel("Age of Player")

plt.ylabel("Potential of Player")

plt.plot(x_train, regressor.predict(x_train),color="blue") # To draw line of regression

plt.show()
# Visualising test dataset

plt.scatter(x_test,y_test,color="red")

plt.xlabel("Age of Player")

plt.ylabel("Potential of Player")

plt.plot(x_train, regressor.predict(x_train),color="blue")

plt.show()
# Finding intercept of linear regression line

regressor.intercept_
# Finding coefficient of linear regression line

regressor.coef_
# Finding mean squared error of linear regression model

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)
# independent variables are - Age, Agility, Balance, stamina, Strength, Composure

x=data.iloc[:,[3,66,68,71,72,79]]
x.head()
# checking if there are null values in x and then filling them. 

x.isnull().any()
x=x.fillna(method='ffill')
x.isnull().any()
# dependent variable = Potential

y=data.iloc[:,8]
y.head()
y.isnull().any()
sns.lineplot(x="Potential", y="Age",data=data,label="Age", ci= None)

sns.lineplot(x="Potential", y="Agility",data=data,label="Agility", ci= None)

sns.lineplot(x="Potential", y="Balance",data=data,label="Balance", ci= None)

sns.lineplot(x="Potential", y="Stamina",data=data,label="Stamina", ci= None)

sns.lineplot(x="Potential", y="Strength",data=data,label="Strength", ci= None)

sns.lineplot(x="Potential", y="Composure",data=data,label="Composure", ci= None)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.predict(x_test)
# Visualising Actual and predicted values of Potential of player

plt.scatter(y_test,y_pred)

plt.xlabel("Actual Potential")

plt.ylabel("Predicted Potential")

plt.show()
regressor.intercept_
regressor.coef_
# let us take the significance level (SL)= 0.05

import statsmodels.formula.api as sm
# fitting all variables in the model

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
# Finding statistical summary of all variables

regressor_OLS.summary()
# independent variable= age

x=data.iloc[:,3]
x.head()
# dependent variable = potential

y=data.iloc[:,8]
y.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train=np.array(x_train)

y_train=np.array(y_train)
x_train=x_train.reshape(-1,1)

y_train=y_train.reshape(-1,1)
lin_reg_1=LinearRegression()
lin_reg_1.fit(x_train,y_train)
x_test=np.array(x_test)
x_test=x_test.reshape(-1,1)
y_pred_1=lin_reg_1.predict(x_test)
# Making polynomail regression model

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x=np.array(x)
x=x.reshape(-1,1)
# Making polynomial matrix of x of degree 3

x_poly=poly_reg.fit_transform(x)
x_poly
x_poly_train,x_poly_test,y_train,y_test=train_test_split(x_poly,y,test_size=0.2, random_state=42)
# Making another object to fit polynomial set

lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly_train,y_train)
y_pred_2=lin_reg_2.predict(x_poly_test)
# Visualizing Linear Regression Model

plt.scatter(x_test,y_test,color='red')

plt.xlabel("Age of Player")

plt.ylabel("Potential of Player")

plt.title("Linear Regression Curve ")

plt.plot(x_train,lin_reg_1.predict(x_train),color='blue')

plt.show()
# Visualizing Polynomial Regression Model

plt.scatter(x_test,y_test,color='red')

plt.xlabel("Age of Player")

plt.ylabel("Potential of Player")

plt.title("Polynomial Regression Curve ")

plt.plot(x_train,lin_reg_2.predict(poly_reg.fit_transform(x_train)),color='blue')

plt.show()
mean_squared_error(y_test,y_pred_2)