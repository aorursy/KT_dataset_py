# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/insurance.csv',skiprows=list(range(0,5)),

                      header=None,

                      names=['Claims','Total_Payment'])
dataset.head()
plt.scatter(dataset.Claims,dataset.Total_Payment)
X = dataset[['Claims']]

y = dataset['Total_Payment']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=13/63, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

model = regressor.fit(X_train,y_train)
y_pred = model.predict(X_test)
from math import sqrt

from sklearn.metrics import mean_squared_error

error = mean_squared_error(y_test,y_pred)

sqrt(error)
plt.scatter(X_train,y_train)

plt.plot(X_train,model.predict(X_train),color='red')