# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
household_income = pd.read_csv("../input/median-household-income-in-the-united-states.csv")
household_income
household_income["year"] = [i for i in range(1984, 2018)]
household_income.plot(kind="scatter", x="year", y="value")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = np.c_[household_income["year"]]
y = np.c_[household_income["value"]]
model.fit(X, y)
X_new = [[2050]]
print(model.predict(X_new))
y_pred = model.predict(X)
plt.plot(X, y, "b:", label="actual values")
plt.plot(X, y_pred, "r-", label="predicted values")
plt.legend()
plt.show()
