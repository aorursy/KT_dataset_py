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
import seaborn as sns;

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import numpy as np



prices=pd.read_csv("/kaggle/input/homeprices.csv")

prices

print("## total no of houses: "+str(len(prices)))
## Check the missing values

prices.info()
prices.isnull().sum()
sns.pairplot(prices)
sns.heatmap(prices.corr(),linewidth=2,vmax=1.0,square=True,linecolor='Grey',annot=True)
#Splitinf Features and lables



X = prices.iloc[:, 0:1].values 

y = prices.iloc[:, 3].values 



# Polynomial Regression



plreg=PolynomialFeatures(degree=12)

xploy=plreg.fit_transform(X)

linreg=LinearRegression()

linreg.fit(xploy,y)

print("Accuarcy is",lin1.score(X,y))
linreg.coef_
linreg.intercept_

print("X shape is"+str(len(xploy)))

print("Y shape is "+str(len(linreg.predict(xploy))))
# Linear Regression

lin1=LinearRegression()

lin1.fit(X,y)

print("Accuarcy is",lin1.score(X,y))
plt.scatter(X, y, color = 'blue') 

  

plt.plot(X, linreg.predict(plreg.fit_transform(X)), color = 'red') 

plt.title('Ploynomial Regression') 

plt.xlabel('Area') 

plt.ylabel('Prices') 

  

plt.show() 

plt.scatter(X, y, color = 'blue') 

  

plt.plot(X, lin1.predict(X), color = 'red') 

plt.title('Linear Regression') 

plt.xlabel('Area') 

plt.ylabel('Prices') 

  

plt.show()
#Mean square error

metrics.mean_squared_error(y,lin1.predict(X))

#Root Mean square error

np.sqrt(metrics.mean_squared_error(y,lin1.predict(X)))
#Sample test Predict

Xtest=[[2600]]

lin1.predict(Xtest)

#Predicted value+Root Mean square error ~ Actual Value

lin1.predict(Xtest)+43114.16896349795