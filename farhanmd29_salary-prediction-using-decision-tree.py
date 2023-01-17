# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

dataset = pd.read_csv('../input/Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, -1].values
X
y
# Fitting the Decision Regression Model to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X, y)
# Predicting a new result

y_pred = regressor.predict(6.5)
y_pred
# Visualising the Regression results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (Regression Model)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
#From the visualization above we can observe that the point on each level is the result of average of the total data points in each interval.Hence the salary is 150000 for the levels ranging in between 5.5 and 6.5.