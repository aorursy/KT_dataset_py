# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the Dataset



dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')



X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:,2].values
'''Splitting Dataset into Training and Test set'''

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Since Data is small, so whole model is needed
# Fitting SVR to Dataset



regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X, y)



# Predicting a new result



y_pred = regressor.predict([[6.5]])

print(y_pred)



# Visualizing the Decision Tree Regression Results



X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color='red')

plt.plot(X_grid, regressor.predict(X_grid), color='blue')

plt.title('Truth or Bluff Decision Tree Regression')

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.show()