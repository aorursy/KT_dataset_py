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
data_path = '../input/canada-per-capita-income-single-variable-data-set/canada_per_capita_income.csv'

df = pd.read_csv(data_path)

df.head()
%matplotlib inline

plt.xlabel('year')

plt.ylabel('per capita income(US$)')

plt.scatter(df.year, df[['per capita income (US$)']], color='red', marker='+')
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(df[['year']], df[['per capita income (US$)']])
reg.predict([[2020]])
%matplotlib inline

plt.xlabel('year', fontsize=20)

plt.ylabel('per capita income (US$)', fontsize=20)

plt.scatter(df.year, df[['per capita income (US$)']], color='red', marker='+')

plt.plot(df.year, reg.predict(df[['year']]), color='blue')
#Let's see how the model predicted the price of Canada's per capita income for the year 2020

m = reg.coef_

b = reg.intercept_



print("This is the coeficient of the model: " + str(m))

print("This is the y-int of the model: " + str(b))
#now let's assign the x-value which is the year we want to predict

x = 2020

y = m*x+b

print("This is the predicted income per capita for the year 2020: " + str(y))

#as we can see, the predicted output from the linear function is the same as the output from the linear regression model