# Feb 19, 2019

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



input_data = "../input/hr_data.csv"



hr_data =pd.read_csv(input_data)



hr_data
hr_data.describe()

# Minimum value for satisfaction_level

hr_data['satisfaction_level'].min()

# Maximum value for satisfaction_level

hr_data['satisfaction_level'].max()
# Separate our data into dependent (Y) and independent(X) variables

X_data = hr_data[['last_evaluation','number_project','average_montly_hours','time_spend_company']]

Y_data = hr_data['satisfaction_level']
# 70/30 Train Test Split

# We will split the data using a 70/30 split. i.e. 70% of the data will be randomly chosen to 

# train the model and 30% will be used to evaluate the model



from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)



X_train.shape

y_train.shape

# Import linear model package (has several regression classes)

from sklearn import linear_model



# Create an instance of linear regression model

reg = linear_model.LinearRegression()

reg.fit(X_train,y_train)



reg.coef_



X_train.columns
# Intercept value

reg.intercept_

print('___ Regression Coefficient ___')

pd.DataFrame(reg.coef_, index=X_data.columns, columns=['Coefficients'])
# Formula y = B0 + B1 + E

# satisfaction _level = 0.62 + 0.265541 * last_evaluation + (-0.041339 * number_project) + 

# (0.000035 * average_montly_hours) + (-0.014081 * time_spend_company)

test_predicted = reg.predict(X_test)

test_predicted
next_data = X_test.copy()



next_data['satisfaction_level'] = y_test

next_data['predicted_satisfaction_level'] = test_predicted

next_data.head(8)
error = next_data['satisfaction_level'] * 100 - next_data['predicted_satisfaction_level'] * 100

error.abs().mean()
(error * error).mean
from sklearn.metrics import mean_absolute_error, r2_score 

from sklearn.metrics import mean_squared_error
mean_squared_error(next_data['satisfaction_level'] * 100, next_data['predicted_satisfaction_level'] * 100)
# Function for square root

np.sqrt(589.5609508936939)
print('R-squared value : %f' % np.sqrt(589.5609508936939))

print('Using score model : %f' % reg.score(X_test, y_test))

print('Using r2score model : %f' %  r2_score(y_test, test_predicted))