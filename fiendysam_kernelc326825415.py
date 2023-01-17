# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Combined_Clean.csv")
data.describe()
data.head()
for i in ["Region", "State", "LandCategory", "Region or State"]:

    data.groupby([i, "Year"])["Acre Value"].sum().unstack().transpose().plot(figsize=(15,10))
from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



models = [GradientBoostingRegressor(), LinearRegression(), BayesianRidge()]
data2 = pd.get_dummies(data)

X = data2.drop("Acre Value", axis = 1)

y = data2["Acre Value"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
for model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(model.__class__)

    print(model.score(X_test, y_test))

    print(mean_absolute_error(y_test, y_pred))