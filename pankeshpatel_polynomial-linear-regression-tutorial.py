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



import matplotlib.pyplot as plt 

import numpy as np 

import seaborn as sns 

plt.style.use('fivethirtyeight')  

import warnings

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook
# We can see that the dataset has 10 levels and the corresponding salary paid to the employee

dataset = pd.read_csv("../input/Position_Salaries.csv")

dataset 
# For the features we are selecting all the rows of column Level 

# represented by column position 1 or -1 in the data set.

X=dataset.iloc[:,1:2].values  



# for the target we are selecting only the salary column which 

#can be selected using -1 or 2 as the column location in the dataset

y=dataset.iloc[:,2].values   

X
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(X,y)
from sklearn.preprocessing import PolynomialFeatures

poly_reg2=PolynomialFeatures(degree=2)

X_poly=poly_reg2.fit_transform(X)

lin_reg_2=LinearRegression()

lin_reg_2.fit(X_poly,y)
poly_reg3=PolynomialFeatures(degree=3)

X_poly3=poly_reg3.fit_transform(X)

lin_reg_3=LinearRegression()

lin_reg_3.fit(X_poly3,y)
plt.scatter(X,y,color='red')

plt.plot(X,lin_reg.predict(X),color='blue')

plt.title('Truth Or Bluff (Linear Regression)')

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.show()
plt.scatter(X,y,color='red')

plt.plot(X,lin_reg_2.predict(poly_reg2.fit_transform(X)),color='blue')

plt.plot(X,lin_reg_3.predict(poly_reg3.fit_transform(X)),color='green')

plt.title('Truth Or Bluff (Polynomial Linear Regression)')

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.show()
X_grid=np.arange(min(X),max(X),0.1) # This will give us a vector.We will have to convert this into a matrix 

X_grid=X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color='red')

plt.plot(X_grid,lin_reg_3.predict(poly_reg3.fit_transform(X_grid)),color='blue')

#plt.plot(X,lin_reg_3.predict(poly_reg3.fit_transform(X)),color='green')

plt.title('Truth Or Bluff (Polynomial Linear Regression)')

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.show()
lin_reg.predict([[6.5]])  # We are assuming the level of the employee is 6.5
lin_reg_2.predict(poly_reg2.fit_transform([[6.5]]))
lin_reg_3.predict(poly_reg3.fit_transform([[6.5]]))