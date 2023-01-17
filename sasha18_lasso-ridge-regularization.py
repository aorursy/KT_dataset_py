#Import numerical libraries

import pandas as pd

import numpy as np



#Import graphical plotting libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Import Linear Regression Machine Learning Libraries

from sklearn import preprocessing

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import r2_score
data = pd.read_csv('../input/carmpg/car-mpg (1).csv')

data.head()
#Drop car name

#Replace origin into 1,2,3.. dont forget get_dummies

#Replace ? with nan

#Replace all nan with median



data = data.drop(['car_name'], axis = 1)

data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})

data = pd.get_dummies(data,columns = ['origin'])

data = data.replace('?', np.nan)

data = data.apply(lambda x: x.fillna(x.median()), axis = 0)
data.head()
X = data.drop(['mpg'], axis = 1) # independent variable

y = data[['mpg']] #dependent variable
#Scaling the data



X_s = preprocessing.scale(X)

X_s = pd.DataFrame(X_s, columns = X.columns) #converting scaled data into dataframe



y_s = preprocessing.scale(y)

y_s = pd.DataFrame(y_s, columns = y.columns) #ideally train, test data should be in columns
#Split into train, test set



X_train, X_test, y_train,y_test = train_test_split(X_s, y_s, test_size = 0.30, random_state = 1)

X_train.shape
#Fit simple linear model and find coefficients

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)



for idx, col_name in enumerate(X_train.columns):

    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))

    

intercept = regression_model.intercept_[0]

print('The intercept is {}'.format(intercept))
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff



ridge_model = Ridge(alpha = 0.3)

ridge_model.fit(X_train, y_train)



print('Ridge model coef: {}'.format(ridge_model.coef_))

#As the data has 10 columns hence 10 coefficients appear here    
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff



lasso_model = Lasso(alpha = 0.1)

lasso_model.fit(X_train, y_train)



print('Lasso model coef: {}'.format(lasso_model.coef_))

#As the data has 10 columns hence 10 coefficients appear here   
#Model score - r^2 or coeff of determinant

#r^2 = 1-(RSS/TSS) = Regression error/TSS 





#Simple Linear Model

print(regression_model.score(X_train, y_train))

print(regression_model.score(X_test, y_test))



print('*************************')

#Ridge

print(ridge_model.score(X_train, y_train))

print(ridge_model.score(X_test, y_test))



print('*************************')

#Lasso

print(lasso_model.score(X_train, y_train))

print(lasso_model.score(X_test, y_test))
#poly = PolynomialFeatures(degree = 2, interaction_only = True)



#Fit calculates u and std dev while transform applies the transformation to a particular set of examples

#Here fit_transform helps to fit and transform the X_s

#Hence type(X_poly) is numpy.array while type(X_s) is pandas.DataFrame 

#X_poly = poly.fit_transform(X_s)

#Similarly capture the coefficients and intercepts of this polynomial feature model
data_train_test = pd.concat([X_train, y_train], axis =1)

data_train_test.head()
import statsmodels.formula.api as smf

ols1 = smf.ols(formula = 'mpg ~ cyl+disp+hp+wt+acc+yr+car_type+origin_america+origin_europe+origin_asia', data = data_train_test).fit()

ols1.params
print(ols1.summary())
#Lets check Sum of Squared Errors (SSE) by predicting value of y for test cases and subtracting from the actual y for the test cases

mse  = np.mean((regression_model.predict(X_test)-y_test)**2)



# root of mean_sq_error is standard deviation i.e. avg variance between predicted and actual

import math

rmse = math.sqrt(mse)

print('Root Mean Squared Error: {}'.format(rmse))
# Is OLS a good model ? Lets check the residuals for some of these predictor.



fig = plt.figure(figsize=(10,8))

sns.residplot(x= X_test['hp'], y= y_test['mpg'], color='green', lowess=True )





fig = plt.figure(figsize=(10,8))

sns.residplot(x= X_test['acc'], y= y_test['mpg'], color='green', lowess=True )
# predict mileage (mpg) for a set of attributes not in the training or test set

y_pred = regression_model.predict(X_test)



# Since this is regression, plot the predicted y value vs actual y values for the test data

# A good model's prediction will be close to actual leading to high R and R2 values

#plt.rcParams['figure.dpi'] = 500

plt.scatter(y_test['mpg'], y_pred)