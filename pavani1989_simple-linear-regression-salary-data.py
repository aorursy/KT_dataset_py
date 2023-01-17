# importing packages

import numpy as np # to perform calculations 

import pandas as pd # to read data

import matplotlib.pyplot as plt # to visualise
#Loading the salary dataset

data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Linear_Regression_Introduction/master/Salary_Data.csv')
#Let's have a look at what our data is like

data.head()
# Scatter plot helps in visualising the data distribution

plt.plot(data.YearsExperience, data.Salary,'rx')
X = data[['YearsExperience']]

y = data['Salary']
# import SK Learn train test split

from sklearn.model_selection import train_test_split 



# Assign variables to capture train test split output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()  # create object for the class



#fit model to our training data i.e learn coefficients

linear_regressor.fit(X_train, y_train)
# prints y-intercept

print(linear_regressor.intercept_)



# prints the coefficient

print(linear_regressor.coef_)
y_pred = linear_regressor.predict(X_test)  # make predictions
plt.plot(X_test, y_test,'rx')

plt.plot(X_test, y_pred, color='black')

plt.show()
# import metrics library

from sklearn import metrics



# print result of MAE

print(metrics.mean_absolute_error(y_test, y_pred))



#print result of MSE

print(metrics.mean_squared_error(y_test, y_pred))



#print result of RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))