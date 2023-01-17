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
import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import pylab as pl

import numpy as np

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline
data1 = pd.read_csv("../input/simple-linear-regression/datasets_10624_14867_Salary_Data.csv")

data2 = pd.read_csv("../input/simple-linear-regression/datasets_8834_12327_data.csv")
data1
data2
data1.describe()



      
data2.describe()
print(data1.shape) 

print(data2.shape)
cd = data1[['YearsExperience','Salary']]

cd.head(9)
cdf = data2[['Height','Weight']]

cdf.head(9)
histogram = cd[['YearsExperience','Salary']]

histogram.hist()

plt.show()
viz = cdf[['Height','Weight']]

viz.hist()

plt.show()
plt.scatter(cd.YearsExperience, cd.Salary,  color='blue')

plt.xlabel("YearsExperience")

plt.ylabel("Salary")

plt.show()
plt.scatter(cdf.Height, cdf.Weight,  color='blue')

plt.xlabel("Height")

plt.ylabel("Weight")

plt.show()
msk_salary = np.random.rand(len(data1)) < 0.8

train_set = cd[msk_salary]

test_set = cd[~msk_salary]
msk = np.random.rand(len(data2)) < 0.8

train = cdf[msk]

test = cdf[~msk]
plt.scatter(train_set.YearsExperience, train_set.Salary,  color='blue')

plt.xlabel("YearsExperience")

plt.ylabel("Salary")

plt.show()
plt.scatter(train.Height, train.Weight,  color='blue')

plt.xlabel("Height")

plt.ylabel("Weight")

plt.show()
from sklearn import linear_model

regr = linear_model.LinearRegression()

train_set_x = np.asanyarray(train_set[['YearsExperience']])

train_set_y = np.asanyarray(train_set[['Salary']])

regr.fit (train_set_x, train_set_y)

# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['Height']])

train_y = np.asanyarray(train[['Weight']])

regr.fit (train_x, train_y)

# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
plt.scatter(train_set.YearsExperience, train_set.Salary,  color='blue')

plt.plot(train_set_x, regr.coef_[0][0]*train_set_x + regr.intercept_[0], '-r')

plt.xlabel("YearsExperience")

plt.ylabel("Salary")
plt.scatter(train.Height, train.Weight,  color='blue')

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')

plt.xlabel("Height")

plt.ylabel("Weight")
from sklearn.metrics import r2_score



test_set_x = np.asanyarray(test_set[['YearsExperience']])

test_set_y = np.asanyarray(test_set[['Salary']])

test_set_y_hat = regr.predict(test_set_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_set_y_hat - test_set_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_set_y_hat - test_set_y) ** 2))

print("R2-score: %.2f" % r2_score(test_set_y_hat , test_set_y) )
from sklearn.metrics import r2_score



test_x = np.asanyarray(test[['Height']])

test_y = np.asanyarray(test[['Weight']])

test_y_hat = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )