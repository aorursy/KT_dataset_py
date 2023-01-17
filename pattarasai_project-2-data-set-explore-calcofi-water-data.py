# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model as lm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/bottle.csv')

data.head()
dataOfProject = data[["Depthm", "T_degC", "Salnty", "O2ml_L", "PO4q", "SiO3qu","NO3q","NH3q","NO2q"]]

dataOfProject.corr()
dataSet = pd.DataFrame()

dataSet["T_degC"] = data["T_degC"]

dataSet["Salnty"] = data["Salnty"]

print(dataSet)
dataSet.describe()
dataSet.isnull().sum()
dataSet = dataSet.dropna()
dataSet.isnull().sum()
dataSet.corr()
plt.scatter(dataSet["Salnty"], dataSet["T_degC"], alpha=0.1)
salnty = dataSet["Salnty"]

salnty = salnty.values.reshape(-1, 1)

print(salnty)
t_degC = dataSet["T_degC"]

t_degC = t_degC.values.reshape(-1, 1)

print(t_degC)
model = lm.LinearRegression()

model.fit(salnty, t_degC)

print(model.intercept_, model.coef_)
predictedValue = model.predict(salnty)

print(predictedValue)
plt.scatter(salnty,t_degC)

plt.plot(salnty,predictedValue, color="blue")

plt.show()
square_error = ((predictedValue - t_degC) ** 2)

print(square_error.sum())
data.boxplot(column=["T_degC","Salnty","O2ml_L","STheta", "T_prec"])

data.boxplot(column=["O2Sat"])
data.corr()
plt.scatter(data["Depthm"], data["T_degC"], alpha=0.1)
plt.scatter(data["Depthm"], data["Salnty"], alpha=0.1)
plt.scatter(data["T_degC"], data["O2ml_L"], alpha=0.1)
plt.scatter(data["Depthm"], data["O2Sat"], alpha=0.1)