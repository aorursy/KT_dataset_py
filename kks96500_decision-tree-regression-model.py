#importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#importing dataset

dataset=pd.read_csv('../input/Position_Salaries.csv')

X=dataset.iloc[:,1:2].values

y=dataset.iloc[:,2].values
#fitting the Decision tree regression model

from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor()

regressor.fit(X,y)
#predicting the new results 

y_pred=regressor.predict([[6.5]])
y_pred
#visualizing the results of Decision Tree Regression model(higher resolution)

X_grid = np.arange(min(X),max(X),0.01)

X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color='red')

plt.plot(X_grid,regressor.predict(X_grid),color='blue')

plt.title('Decision Tree regression Model')

plt.xlabel('Position')

plt.ylabel('salary')

plt.show()