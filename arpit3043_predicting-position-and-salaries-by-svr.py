# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# for plotting Graphs

import matplotlib.pyplot as plt
# Data Pre-Processing

dataset = pd.read_csv("/kaggle/input/position-salary-dataset/Position_Salaries.csv")

X = dataset.iloc[:,1:-1].values

y = dataset.iloc[:,-1].values
X
y
y = y.reshape(len(y),1)

y
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X = sc_X.fit_transform(X)

y = sc_y.fit_transform(y)
X
y
# Training the SVR model on the whole dataset

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X,y)
# Predicting a new result

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
# Visualising the SVR results

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')

plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')

plt.title('Truth or Bluff (SVR)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
# Visualising the SVR results (for higher resolution and smoother curve)

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')

plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')

plt.title('Truth or Bluff (SVR)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()