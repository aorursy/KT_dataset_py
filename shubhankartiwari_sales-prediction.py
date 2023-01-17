# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

import statsmodels.api as sm
data = pd.read_csv("../input/Advertising.csv")
data.head()
data = data.drop(['Unnamed: 0'],axis = 1)
data.head()
plt.figure(figsize = (16,8))

plt.scatter(

    data['TV'],

    data['sales'],

    c = 'red'

)

plt.xlabel("Money spent on TV")

plt.ylabel("Sales")

plt.show()
X = data['TV'].values.reshape(-1,1)

y = data['sales'].values.reshape(-1,1)

reg = LinearRegression()

reg.fit(X,y)

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0],reg.coef_[0] [0]))

predictions = reg.predict(X)

plt.figure(figsize = (16,8))

plt.scatter(

    data['TV'],

    data['sales'],

    c = 'red'

)

plt.plot(

    data['TV'],

    predictions,

    c = 'green'

)

plt.xlabel("Money spent on TV")

plt.ylabel('Sales')

plt.show()
X = data['TV']

y = data['sales']

X2 = sm.add_constant(X)

est = sm.OLS(y,X2)

es2 = est.fit()

print(es2.summary())
plt.figure(figsize = (16,8))

plt.scatter(

    data['radio'],

    data['sales'],

    c = 'green'

)

plt.xlabel("Money spent on radio")

plt.ylabel("Sales")

plt.show()
X = data['radio'].values.reshape(-1,1)

y = data['sales'].values.reshape(-1,1)

reg = LinearRegression()

reg.fit(X,y)

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0],reg.coef_[0] [0]))
predictions = reg.predict(X)

plt.figure(figsize = (16,8))

plt.scatter(

    data['radio'],

    data['sales'],

    c = 'green'

)

plt.plot(

    data['radio'],

    predictions,

    c = 'blue'

)

plt.xlabel("Money spent on radio")

plt.ylabel('Sales')

plt.show()
X = data['radio']

y = data['sales']

X2 = sm.add_constant(X)

est = sm.OLS(y,X2)

es2 = est.fit()

print(es2.summary())
plt.figure(figsize = (16,8))

plt.scatter(

    data['newspaper'],

    data['sales'],

    c = 'brown'

)

plt.xlabel("Money spent on newspaper")

plt.ylabel("Sales")

plt.show()
X = data['newspaper'].values.reshape(-1,1)

y = data['sales'].values.reshape(-1,1)

reg = LinearRegression()

reg.fit(X,y)

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0],reg.coef_[0] [0]))
predictions = reg.predict(X)

plt.figure(figsize = (16,8))

plt.scatter(

    data['newspaper'],

    data['sales'],

    c = 'brown'

)

plt.plot(

    data['newspaper'],

    predictions,

    c = 'yellow'

)

plt.xlabel("Money spent on radio")

plt.ylabel('Sales')

plt.show()
X = data['newspaper']

y = data['sales']

X2 = sm.add_constant(X)

est = sm.OLS(y,X2)

es2 = est.fit()

print(es2.summary())
plt.figure(figsize = (16,8))

plt.scatter(

    data['TV'],

    data['radio'],

    c = 'violet'

)

plt.xlabel("Money spent on TV")

plt.ylabel("Money spent on radio")

plt.show()
X = data['TV'].values.reshape(-1,1)

y = data['radio'].values.reshape(-1,1)

reg = LinearRegression()

reg.fit(X,y)

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0],reg.coef_[0] [0]))
predictions = reg.predict(X)

plt.figure(figsize = (16,8))

plt.scatter(

    data['TV'],

    data['radio'],

    c = 'violet'

)

plt.plot(

    data['TV'],

    predictions,

    c = 'green'

)

plt.xlabel("Money spent on TV")

plt.ylabel('Money spent on radio')

plt.show()
X = data['TV']

y = data['radio']

X2 = sm.add_constant(X)

est = sm.OLS(y,X2)

es2 = est.fit()

print(es2.summary())
plt.figure(figsize = (16,8))

plt.scatter(

    data['radio'],

    data['newspaper'],

    c = 'orange'

)

plt.xlabel("Money spent on radio")

plt.ylabel("Money spent on newspaper")

plt.show()
X = data['radio'].values.reshape(-1,1)

y = data['newspaper'].values.reshape(-1,1)

reg = LinearRegression()

reg.fit(X,y)

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0],reg.coef_[0] [0]))
predictions = reg.predict(X)

plt.figure(figsize = (16,8))

plt.scatter(

    data['radio'],

    data['newspaper'],

    c = 'orange'

)

plt.plot(

    data['radio'],

    predictions,

    c = 'cyan'

)

plt.xlabel("Money spent on radio")

plt.ylabel('Money spent on newspaper')

plt.show()
X = data['radio']

y = data['newspaper']

X2 = sm.add_constant(X)

est = sm.OLS(y,X2)

es2 = est.fit()

print(es2.summary())