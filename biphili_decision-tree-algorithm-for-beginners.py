# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 

import numpy as np 

import seaborn as sns 

plt.style.use('fivethirtyeight')  

import warnings

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook
dataset=pd.read_csv('../input/Position_Salaries.csv')

dataset
X=dataset.iloc[:,1:2].values  # For the features we are selecting all the rows of column Level represented by column position 1 or -1 in the data set.

y=dataset.iloc[:,2].values    # for the target we are selecting only the salary column which can be selected using -1 or 2 as the column location in the dataset

#X
from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor(random_state=0) #Default criterian is mse mean square error 

regressor.fit(X,y)
y_pred=regressor.predict([[6.5]])

y_pred
plt.scatter(X,y,color='red')

plt.plot(X,regressor.predict(X),color='blue')

plt.title('Truth or BLuff (Decision Tree Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary in $')

plt.show()
X_grid=np.arange(min(X),max(X),0.1)

X_grid=X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color='red')

plt.plot(X_grid,regressor.predict(X_grid),color='blue')

plt.title('Truth or BLuff (Decision Tree Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary in $')

plt.show()
X_grid=np.arange(min(X),max(X),0.01)

X_grid=X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color='red')

plt.plot(X_grid,regressor.predict(X_grid),color='blue')

plt.title('Truth or BLuff (Decision Tree Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary in $')

plt.show()