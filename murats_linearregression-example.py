# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/MonthAndSales.csv", sep=",")

df.info()
df.head()
df.describe()
sales = df[['Sales']]

months = df[['Months']]



x_train, x_test, y_train, y_test = train_test_split(months, sales, random_state=0, test_size=0.33)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)

Y_test = sc.fit_transform(y_test)
model = LinearRegression()

model.fit(X_train, Y_train)

pred = model.predict(X_test)



print(model.score(X_train,Y_train))
model = LinearRegression()

model.fit(x_train, y_train)

pred = model.predict(x_test)



print(model.score(x_train,y_train))
x_train = x_train.sort_index()

y_train = y_train.sort_index()

plt.title("Months vs Sales")

plt.plot(x_train,y_train)

plt.plot(x_test, pred)

plt.xlabel("Months")

plt.ylabel("Sales")

plt.legend()

plt.show()