# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pylab as pl

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/fuelconsumption/FuelConsumptionCo2.csv")



# take a look at the dataset

df.head()
#selecting relavent data

cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]

cdf.head(5)
sns.scatterplot(x = "ENGINESIZE", y = "CO2EMISSIONS", data = cdf, ci = False)
msk = np.random.rand(len(df)) < 0.8

train = cdf[msk]

test = cdf[~msk]
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model

train_x = np.asanyarray(train[['ENGINESIZE']])

train_y = np.asanyarray(train[['CO2EMISSIONS']])



test_x = np.asanyarray(test[['ENGINESIZE']])

test_y = np.asanyarray(test[['CO2EMISSIONS']])





poly = PolynomialFeatures(degree=2)

train_x_poly = poly.fit_transform(train_x)

train_x_poly
clf = linear_model.LinearRegression()

train_y_ = clf.fit(train_x_poly, train_y)

# The coefficients

print ('Coefficients: ', clf.coef_)

print ('Intercept: ',clf.intercept_)
axes = sns.scatterplot(x = "ENGINESIZE", y = "CO2EMISSIONS", data = cdf, ci = False)

XX = np.arange(0.0, 10.0, 0.1)

yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)

plt.plot(XX, yy, '-r' )

plt.xlabel("Engine size")

plt.ylabel("Emission")
from sklearn.metrics import r2_score



test_x_poly = poly.fit_transform(test_x)

test_y_ = clf.predict(test_x_poly)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_ , test_y) )