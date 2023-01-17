import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
sal_data = pd.read_csv("../input/salary/Salary.csv") 
sal_data.head()
X=sal_data["YearsExperience"]

Y = sal_data['Salary']

X=X.values.reshape(-1,1)
X.shape
Y.shape
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X, Y)
#predicting a new value



regressor.predict([[9.3]])
#predicting a new value



regressor.predict([[10.8]])
accuracy = regressor.score(X,Y)

print(accuracy*100,'%')
#Visualising the Random Forest Regression results



X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Predicting salary using Random Forest Regression ')

plt.xlabel('Years of experience')

plt.ylabel('Salary')

plt.show()