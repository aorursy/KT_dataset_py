# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('../input/Position_Salaries.csv')
dataset.head()
# Visualising the Regression results (for higher resolution and smoother curve)
def plot(X, regressor):
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (Regression Model)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
X = dataset.iloc[:, 1:2].values.reshape(-1, 1)
y = dataset.iloc[:, 2].values
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])
plot(X,regressor)
y_pred
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])
plot(X,regressor)
y_pred
