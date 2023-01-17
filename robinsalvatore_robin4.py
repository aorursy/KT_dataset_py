# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/measurements.csv')
df
df.describe()
cdf = df[['AC','rain','sun','temp_inside','temp_outside','speed','consume','distance']]
cdf.head(9)
viz = cdf[['AC','rain','sun','temp_inside','temp_outside','speed','consume','distance']]
viz.hist()
plt.show()
plt.scatter(cdf.speed, cdf.consume,color='blue')
plt.xlabel("speed")
plt.ylabel("consume")
plt.show()
viz = cdf[['AC','rain','sun','temp_inside','temp_outside','speed','consume','distance']]
viz.hist()
plt.show()
plt.scatter(cdf.speed, cdf.consume,color='blue')
plt.xlabel("speed")
plt.ylabel("consume")

msk = np.random.rand(len(df)) < 0.8
car = cdf[msk]
test = cdf[~msk]
plt.scatter(cdf.speed,cdf.rain,color="blue")
plt.xlabel("speed")
plt.ylabel("rain")
plt.show()
from sklearn import linear_model
regr = linear_model.LinearRegression()
car_x = np.asanyarray(car[['rain']])
car_y = np.asanyarray(car[['sun']])
regr.fit (car_x, car_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
plt.scatter(car.rain, car.sun,  color='blue')
plt.plot(car_x, regr.coef_[0][0]*car_x + regr.intercept_[0], '-r')
plt.xlabel("rain")
plt.ylabel("sun")
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['sun']])
test_y = np.asanyarray(test[['rain']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )