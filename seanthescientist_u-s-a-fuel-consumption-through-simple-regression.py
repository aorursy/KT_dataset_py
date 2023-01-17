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

import pandas as pd

import pylab as pl

import numpy as np

%matplotlib inline
df = pd.read_excel('/kaggle/input/consumpusa/_fuel_consump.xlsx')

df.head()
df.shape
df.columns
df.tail()
us=df.rename(columns={"# Cyl": "Cylinders","Eng Displ": "Engine_L", "Comb CO2 Rounded Adjusted (as shown on FE Label)": "Comb_CO2", "Comb FE (Guide) - Conventional Fuel":"Comb_FE"})

us.head()
usdf = us[['Cylinders', 'Engine_L', 'Comb_CO2', 'Comb_FE']]

usdf
viz = usdf[['Cylinders','Engine_L','Comb_CO2','Comb_FE']]

viz.hist()

plt.show()
plt.scatter(usdf.Comb_FE, usdf.Comb_CO2,  color='purple')

plt.xlabel("Combined Fuel Economy")

plt.ylabel("Combined Emission")

plt.show()
plt.scatter(usdf.Cylinders, usdf.Comb_CO2,  color='blue')

plt.xlabel("Engine Size")

plt.ylabel("Combined CO2 Emissions")



plt.show()
plt.scatter(usdf.Engine_L, usdf.Comb_CO2,  color='green')

plt.xlabel("Engine Size")

plt.ylabel("Combined CO2 Emissions")



plt.show()
msk = np.random.rand(len(df)) < 0.8

train = usdf[msk]

test = usdf[~msk]
plt.scatter(train.Engine_L, train.Comb_CO2,  color='blue')

plt.xlabel("Engine Size")

plt.ylabel("Emission")

plt.show()
from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['Engine_L']])

train_y = np.asanyarray(train[['Comb_CO2']])

regr.fit (train_x, train_y)

# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
plt.scatter(train.Engine_L, train.Comb_CO2,  color='blue')

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')

plt.xlabel("Engine Size")

plt.ylabel("Emissions")
from sklearn.metrics import r2_score



test_x = np.asanyarray(test[['Engine_L']])

test_y = np.asanyarray(test[['Comb_CO2']])

test_y_hat = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )