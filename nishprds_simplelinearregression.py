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
# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#print(train.head(2))

# Remove the duplicate record. 
train = train.drop_duplicates()
test = test.drop_duplicates()
#Data Cleaning Process
# check if any columns in data set is blank
print("Does training data have null values: ",train.isnull().any().any()) 
print("Does test data have null values: ",test.isnull().any().any())

# To check which column has NaN
print("Column with NA values: ",train.columns[train.isna().any()].tolist())

#To check number of non null values
train.info()

# Since just one row in y is NA we can drop the row
train=train.dropna()
# Replace NaN values with mean for training data
#train["y"].fillna(train["y"].mean(), inplace=True)

print("Does training data have null values after cleaning: ",train.isnull().any().any()) 
train_x = np.asanyarray(train[['x']])
train_y = np.asanyarray(train['y'])
#print(train_x)
#print(train_y)
test_x = np.asanyarray(test[['x']])
test_y = np.asanyarray(test['y'])
#print(test.head(2))
#print(test_x)
#print(test_y)
print("Train set: ", train_x.shape, train_y.shape)
print("Test set: ", test_x.shape, test_y.shape)
#Plotting the training set for checking correlation between x and y
plt.scatter(train.x, train.y,color='blue')
plt.xlabel("X")
plt.ylabel("Y")
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_x,train_y)
#Coefficients
print("Coefficient ", regressor.coef_)
print("Intercept ",regressor.intercept_)
# Visualising the Training set results : Using formula
plt.scatter(train_x, train_y, color='blue')
plt.plot(train_x, regressor.intercept_ + regressor.coef_[0] * train_x,  color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
# Predicting the Test set results
y_pred = regressor.predict(test_x)
# Visualising the Test set results
plt.scatter(test_x, test_y, color = 'green')
plt.plot(train_x, regressor.predict(train_x), color = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# To check how fit is the model
from sklearn.metrics import r2_score
print("Mean absolute error: %0.2f" %np.mean(np.absolute(y_pred, test_y)))
print("Residual sum of errors (MSE) %0.2f" %np.mean((y_pred - test_y)**2))
print("R2-square : %.2f" %r2_score(y_pred,test_y))