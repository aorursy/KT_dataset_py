# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt # For plotting the visualizations
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/Position_Salaries.csv')
#Displaying the dataset
dataset
#The problem statement is that the candidate with level 6.5 had a previous salary of 160000. In order to hire him in our new company we would like to confirm if he is being honest about his last salary and we will can predict this using the Random Forest method used in this notebook.
# Splitting the dataset 
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
X
y
# Fitting the Regression Model to the dataset with 10 tree prediction
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X, y)
# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# Predicting a new result
y_pred = regressor.predict(6.5)
y_pred
#Now we will add more trees to improve the accuracy of the prediction
# Fitting the Regression Model to the dataset with 10 tree prediction
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(X, y)
# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# Predicting a new result
y_pred = regressor.predict(6.5)
y_pred
#This is a true accurate prediction of the previous salary of the employee. Cheers!