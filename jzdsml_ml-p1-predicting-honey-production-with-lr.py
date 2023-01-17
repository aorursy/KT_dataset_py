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
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import numpy as np



temperature = np.array(range(60, 100, 2))

print(temperature)

temperature = temperature.reshape(-1, 1) #reshape for fitting

print(temperature)



sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]
line_fitter = LinearRegression()

line_fitter.fit(temperature, sales)
sales_predict = line_fitter.predict(temperature)
plt.figure()

plt.plot(temperature, sales, 'o')

plt.plot(temperature, sales_predict, '-')

plt.show()
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn import linear_model



# read csv file of honey production in the usa 1998-2012

df = pd.read_csv("../input/honeyproduction.csv")

df.head()
#For now, we care about the total production of honey per year.



prod_per_year = df.groupby("year").totalprod.mean().reset_index()



X = prod_per_year["year"].values.reshape(-1,1)



X
y = prod_per_year["totalprod"].values.reshape(-1,1)



y
regr = linear_model.LinearRegression()

regr.fit(X, y)

print(regr.coef_, regr.intercept_)
ax = plt.subplot(111)

ax.scatter(X, y)

y_predict = regr.predict(X)

plt.plot(X, y_predict, 'red')

plt.title("the total production of honey per year")

plt.xlabel("year")

plt.ylabel("total production")



#Let’s predict what the year 2050 may look like in terms of honey production.



X_future = np.array(range(2013, 2051)).reshape(-1,1)

future_predict = regr.predict(X_future)

ax2 = plt.subplot(111)

ax2.scatter(X, y)

y_predict = regr.predict(X)

plt.plot(X, y_predict, 'red')

plt.title("the total production of honey per year")

plt.xlabel("year")

plt.ylabel("total production")

plt.plot(X_future, future_predict, 'green')

plt.show()