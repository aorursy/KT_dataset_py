# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting and Visualizing data

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
print(data.describe())

# Removing any data point above x = 100
# There is only one record
data = data[data.x <= 100]
# Separating dependednt & Indepented Variables 
x = data.iloc[:, 0:1].values
y = data.iloc[:, 1]
# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)
print(x[:10])
print('\n')
print(y[:10])
# Model Import and Build
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

pred = regressor.predict(x_test)
# Visualization
## Check the fitting on training set
plt.scatter(x_train, y_train)
plt.plot(x_train, regressor.predict(x_train), color='black')
plt.title('Fit on training set')
plt.xlabel('X-Train')
plt.ylabel('Y-Train')
## Check fitting on validation set
plt.scatter(x_test, y_test, color='r')
plt.plot(x_test, pred, color='b')
plt.title('Validation set')
plt.xlabel('X-Test')
plt.ylabel('Y-Test')
## Final test on Test Set
test = pd.read_csv('../input/test.csv')
test.plot.scatter('x', 'y', color='g')
plt.plot(test['x'], regressor.predict(test.iloc[:,0:1].values), color='blue')
plt.title('Linear Regression Ouput on Test Data Set')
plt.xlabel('X-Values')
plt.ylabel('Y-Values')
plt.show()
