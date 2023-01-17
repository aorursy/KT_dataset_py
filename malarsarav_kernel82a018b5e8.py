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
import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
dataset = pd.read_csv('../input/advertising.csv')

print(dataset.shape)

dataset.describe()
## 2D-plot for Money spent on TV Ads and its effect on Sales

dataset.plot(x='TV', y='Sales', style='o')  

plt.title('TV vs Sales')  

plt.xlabel('Money spent on TV Ads')  

plt.ylabel('Sales')  

plt.show()
## 2D-plot for Money spent on Newspaper Ads and its effect on Sales

dataset.plot(x='Newspaper', y='Sales', style='o')  

plt.title('Newspaper vs Sales')  

plt.xlabel('Money spent on Newspaper Ads')  

plt.ylabel('Sales')  

plt.show()
## 2D-plot for Money spent on Radio Ads and its effect on Sales

dataset.plot(x='Radio', y='Sales', style='o')  

plt.title('Radio vs Sales')  

plt.xlabel('Money spent on Radio Ads')  

plt.ylabel('Sales')  

plt.show()
plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(dataset['Sales'])

## The avg seems to lie nearly between 15 to 20
## Basic preprocessing to find whether dataset has noisy data

print(dataset.isna().any())
## Our next step is to divide the data into “attributes” and “labels”. 

## Attributes are the independent variables while labels are dependent variables whose values are to be predicted. 

X = dataset['TV'].values.reshape(-1,1) # Attributes/Features

y = dataset['Sales'].values.reshape(-1,1) # Label
## To split the data into test set(20%) and train set(80%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Simple linear regression model

regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm
## The linear regression model basically finds the best value for the intercept and slope, which results in a line that best fits the data. 

## To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset



#To retrieve the intercept:

print(regressor.intercept_)

#For retrieving the slope:

print(regressor.coef_)



## This means that for every one unit of effect in TV ads, the Effect in the sales is about to increase by 0.054%.
## To predict the test data with our trained model

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df
## Bar graph representation on actual vs predicted by the model

df1 = df

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
## Below Visualization to see the linear fit of our model with actual and prediction.



plt.scatter(X_test, y_test,  color='gray')

plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.show()
## For Regression, there are three metrics to consider which are shown below



## Mean Absolute Error (MAE) is the mean of the absolute value of the errors.

## Mean Squared Error (MSE) is the mean of the squared errors

## Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
## In the above cells , we handled the problem with involving sales and TV data, where as in the real world problems we need to include certain other variables

## for consideration. Linear regression involving multiple variables is called “multiple linear regression” or multivariate linear regression.



## The steps to perform multiple linear regression are almost similar to that of simple linear regression. The difference lies in the evaluation. 

## You can use it to find out which factor has the highest impact on the predicted output and how different variables relate to each other.
## In multiple linear regression, till the preprocessing steps above has to be followed and need to proceed further as shown below.



## Dataset 



X = dataset[['TV','Newspaper','Radio']]

y = dataset['Sales']
## To split the data with multiple independent variables and dependent variable(multi attributes) - Train 80% and Test 20% Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
## Multiple linear model

regressor = LinearRegression()  

regressor.fit(X_train, y_train)
## In the case of multivariable linear regression, the regression model has to find the most optimal coefficients for all the attributes.



coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  

coeff_df



## From the below result which means the following interpretaion



## 1. For 1 unit of TV Ads there is 0.05% increase in Sales

## 2. For 1 unit of Newspaper Ads there is 0.003% of decrease in Sales

## 3. For 1 unit of Radio Ads there is 0.11% of increase in Sales
## Predicting the results with test data after the model has been trained with multiattributes

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1 = df.head(25)

print(df1)
df1.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))





## The RMSE value seems to be lesser than the simple linear regression model which is around 2.5, so this makes more efficient prediction with multidimension 

## than the simple model.





## Alternate for finding RMSE 

from sklearn.model_selection import cross_val_score

MSEs = cross_val_score(regressor, X, y, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)



print("Alternative way of representing error :",mean_MSE)
## Ridge regression for handling our advertising dataset



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge



alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]



ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]} ## 𝜆‖𝑤‖2 - alpha value 

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(X, y)



print(ridge_regressor.best_params_)

print(ridge_regressor.best_score_)
## LASSO regression for handling our advertising dataset



from sklearn.linear_model import Lasso



lasso = Lasso()

parameters = {'alpha': [1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(X, y)

print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_)
