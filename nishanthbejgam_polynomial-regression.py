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
Dataset=pd.read_csv('../input/Salary_Data.csv')
Dataset.shape
Xloc = Dataset.iloc[:,0:1].values

Yloc = Dataset.iloc[:,1:2].values

print(Xloc.shape,Yloc.shape)
from sklearn.preprocessing import PolynomialFeatures

XPolyReg = PolynomialFeatures(degree = 4)

XPoly=XPolyReg.fit_transform(Xloc)
from sklearn.model_selection import train_test_split

XTrain,XTest,YTrain,YTest = train_test_split(XPoly,Yloc,test_size=0.20,random_state=0)

print(XTrain.shape,YTrain.shape)

print(XTest.shape,YTest.shape)
from sklearn.linear_model import LinearRegression

Regressor = LinearRegression()

Regressor.fit(XTrain,YTrain)
YPred = Regressor.predict(XTest)
import matplotlib.pyplot as plt

# Visualising the Polynomial Regression results

plt.scatter(Xloc, Yloc, color = 'red')

plt.plot(Xloc,Regressor.predict(XPolyReg.fit_transform(Xloc)),color = 'blue')

plt.title('Truth or Bluff (Polynomial Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
# Visualising the Polynomial Regression for higher

X_grid=np.arange(min(Xloc), max(Xloc), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(Xloc, Yloc, color = 'red')

plt.plot(X_grid,Regressor.predict(XPolyReg.fit_transform(X_grid)),color = 'blue')

plt.title('Truth or Bluff (Polynomial Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
