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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('../input/New Microsoft Excel Worksheet (2).csv')
dataset.describe
dataset.columns
employees = dataset.iloc[:, 1].values.reshape(-1, 1)  # values converts it into a numpy array

sales = dataset.iloc[:, 2].values.reshape(-1, 1)

branches = dataset.iloc[:, 0].values.reshape(-1,1)



fig = plt.figure()

fig.add_subplot(111)

plt.scatter(employees, sales, alpha=1, c=branches, edgecolors='none', s=30,  label= branches)

plt.xlabel("No of employees")

plt.ylabel("Monthly sales")



linear_regressor = LinearRegression()

linear_regressor.fit(employees, sales)

Y_Pred = linear_regressor.predict(employees)

plt.plot(employees, Y_Pred, color= 'red')



plt.show()

print(linear_regressor.score(employees, sales))

print(linear_regressor.coef_)

print(linear_regressor.intercept_)
print(Y_Pred)