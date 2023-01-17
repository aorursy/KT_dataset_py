import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
regressor.predict([[6.5]])
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
'''
Here we are predicting each and every value of X starting from level 1 till level 10 with an interval of 0.01 to know the outcome and plot each value, as we plot each value we can
see that the model is predicting same outcome for level 1.5 to 2.5 as salary of a level 2, 2.5 to 3.5 as outcome of salary of level 3
If we haven't predicted outcome of each value we will get a graph which is drawn in the next 
code cell and there is very less to interpret from that.
'''
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()