import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

#print(os.listdir("../input")).
data = pd.read_csv('../input/Position_Salaries.csv')

x = data.iloc[:,1:2].values

y = data.iloc[:, 2].values
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(x,y)
# Predicting a new result.

from numpy import array

y_pred = regressor.predict(array(6.5).reshape(-1, 1))
# Visualising the Decision Tree Regression results

x_grid = np.arange(min(x), max(x), 0.01)

x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color = 'red')

plt.plot(x_grid,regressor.predict(x_grid),color='blue')

plt.title('Decision Tree Regression')

plt.xlabel('Position')

plt.ylabel('Salary')

plt.show()