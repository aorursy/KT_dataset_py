# importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory

# print all the file/directories present at the path

import os

print(os.listdir("../input/"))
# importing the dataset

dataset = pd.read_csv('../input/Position_Salaries.csv')
dataset.head()
dataset.info()
plt.plot(dataset.iloc[:,1:-1],dataset.iloc[:,-1],color='red')

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.title('Position Level VS Salary')

plt.show()
# matrix of features as X and dep variable as Y (convert dataframe to numpy array)

X = dataset.iloc[:,1:-1].values          #Level

Y = dataset.iloc[:,-1].values           #Salary
X
# The graph plot is a polynomial curve 



# Applying Polynomial Regression 



from sklearn.preprocessing import PolynomialFeatures

reg = PolynomialFeatures(degree=2)            

# We start with degree = 2 to generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or 

# equal to the specified degree e.g. in our case [1, x, x^2]

poly_X = reg.fit_transform(X)           # poly_X is final matrix of features







poly_X
# Apply the Simple Linear regression to the polynomial matrix of features poly_X



from sklearn.linear_model import LinearRegression

slr = LinearRegression()

slr.fit(poly_X,Y)
# Testing the prediction e.g. on X value = 6.5



y_pred = slr.predict(reg.fit_transform([[6.5]]))





# slr.predict input type ---> X : {array-like, sparse matrix}, shape (n_samples, n_features)

# Hence, created np array of int 6.5 with shape = (1,1) and then convert it to polynomial feature form.



# The precited output would be also scaled value; apply the inverse transformation



#y_pred = sc_Y.inverse_transform(y_pred_featured_scaled)

y_pred
# Testing with poly degree = 3



from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

reg1 = PolynomialFeatures(degree=3)            

poly_X_deg3 = reg1.fit_transform(X)          

slr1 = LinearRegression()

slr1.fit(poly_X_deg3,Y)
y_pred = slr1.predict(reg1.fit_transform([[6.5]]))

y_pred
# Testing with poly degree = 4



from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

reg2 = PolynomialFeatures(degree=4)            

poly_X_deg4 = reg2.fit_transform(X)          

slr2 = LinearRegression()

slr2.fit(poly_X_deg4,Y)
y_pred = slr2.predict(reg2.fit_transform([[6.5]]))

y_pred
plt.plot(dataset.iloc[:,1:-1],dataset.iloc[:,-1],color='red')    # Actual 

plt.plot(X,slr2.predict(reg2.fit_transform(X)),color='blue')     # Predicted one 

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.title('Position Level VS Salary')

plt.show()