import numpy as np 

import pandas as pd # csv read in tool

from sklearn.linear_model import LinearRegression # sk-learn's Linear Regression model

from sklearn.model_selection import train_test_split # sk-learn trainning and testing 

from sklearn import metrics # matrics pakages

import matplotlib.pyplot as plt # plotting tool 

import math # Math pakages

from sklearn.metrics import mean_squared_error
data = pd.read_csv('../input/kc_house_data.csv')

data.head()
data.columns
X = data[['bedrooms','bathrooms','sqft_living','floors','grade','sqft_basement']]

Y = data['price']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 1)
xtrain = np.asmatrix(xtrain)

xtest = np.asmatrix(xtest)

ytrain = np.ravel(ytrain)

ytest = np.ravel(ytest)
plt.scatter(X.sqft_basement, Y)
plt.scatter(X.grade, Y)
plt.scatter(X.bedrooms, Y)
plt.scatter(X.bathrooms, Y)
plt.scatter(X.sqft_living, Y)
plt.scatter(X.floors, Y)
Linearmodel = LinearRegression()

Linearmodel.fit(xtrain, ytrain)



Linearmodel.coef_
pd.DataFrame(list(zip(X.columns, Linearmodel.coef_)))
Linearmodel.predict(np.array([[2, 1, 2500, 2, 7, 0]]))
predicted = Linearmodel.predict(xtrain)

((predicted-ytrain) ** 2).sum()/len(ytrain)
(abs(predicted-ytrain)/ytrain).sum()/len(ytrain)
predicted_test = Linearmodel.predict(xtest)

((predicted_test-ytest) ** 2).sum()/len(ytrain)
(abs(predicted_test-ytest)/ytest).sum()/len(ytest)