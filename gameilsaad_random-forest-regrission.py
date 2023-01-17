import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/position-salariescsv/Position_Salaries.csv')

x=dataset.iloc[:,1:-1].values

y=dataset.iloc[:,-1].values
dataset.head()
print(x)
dataset.tail()

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(x, y)
regressor.predict([[6.5]])
x_grid=np.arange(min(x),max(x),0.01)

x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color='red')

plt.plot(x_grid,regressor.predict(x_grid),color='blue')

plt.title('Random Forest Regression')

plt.xlabel('Position Level')

plt.ylabel('Salay')

plt.show()