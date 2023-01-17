# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing the libraries
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt
#Importing the dataset
dataset = pd.read_csv('../input/Position_Salaries.csv')
dataset.head()
#Assigning the X and y variables from the dataset
X = dataset.iloc[:, 1:2]

y = dataset.iloc[:, 2:]



X.head()
y.head()
#Now fitting the Decision Tree Regression model in the dataset
from sklearn.tree import DecisionTreeRegressor



regressor_dtr = DecisionTreeRegressor(random_state = 0)



regressor_dtr.fit(X, y)

y_predict = regressor_dtr.predict(([[6.5]]))



y_predict
#Visualising the model
X_grid = np.arange(min(X.values), max(X.values), 0.01)



X_grid = X_grid.reshape(len(X_grid), 1)



plt.scatter(X, y , color = 'red')

plt.plot(X_grid,regressor_dtr.predict(X_grid), color = 'blue' )

plt.title('Truth vs bluff (DTR model)')

plt.xlabel('Position')

plt.ylabel('Salary')

plt.show