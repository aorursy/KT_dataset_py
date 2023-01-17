#import the initial libraries and packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#The csv file is loaded as a dtaframe.

dataset = pd.read_csv("/kaggle/input/insurance/insurance.csv")
len(dataset)
dataset.head()
dataset.dtypes
dataset.isnull().sum()
#correlation between columns with respect to charges

dataset.corr()["charges"].sort_values()
#Label Encoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dataset.sex = le.fit_transform(dataset.sex)

dataset.smoker = le.fit_transform(dataset.smoker)

dataset.region = le.fit_transform(dataset.region)
#checking data type

dataset.dtypes
#checking correlation again

dataset.corr()["charges"].sort_values()
#heatmap

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize = (20,10))

sns.heatmap(dataset.corr(), annot = True)
#dividing the dataset into X and y matrix

#X has the features needed to predict charges

#y has the values of charges

X = dataset.iloc[:,0:-1].values

y = dataset.iloc[:,6:7].values
X
y
import statsmodels.api as sm
#creating a new matrix with the first column consisting of all ones

X = np.append(arr = np.ones((1338,1)).astype(int), values = X, axis = 1)
#selecting all the features initially to determine the p value

X_opt = X[:,[0,1,2,3,4,5,6]]
#Using Ordinary Least Square (OLS) to determine the p value and other features

reg_OLS = sm.OLS(y,X_opt).fit()

reg_OLS.summary()
#After removing the x2 column

X_opt = X[:,[0,1,3,4,5,6]]

reg_OLS = sm.OLS(y,X_opt).fit()

reg_OLS.summary()
#reinitializing the matrix X and y

X = dataset.iloc[:,0:-1].values

y = dataset.iloc[:,6:7].values
#split into train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
#checking length of training set

len(X_train)
#checking length of test set

len(X_test)
#importing sklearn for Multivariate Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()



#fit the model with the training set

regressor.fit(X_train, y_train)
#calculating the score of the model

regressor.score(X_test, y_test)
#reinitializing the X and y data 

X = dataset.iloc[:,0:-1].values

y = dataset.iloc[:,6:7].values
#importing the PolynomialFeatures from sklearn.preprocessing

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2)
#fitting the polynomial equation model

X_poly = poly.fit_transform(X)
#split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_poly,y, test_size = 0.2, random_state = 0)
#train the regressor model

regressor.fit(X_train, y_train)
#score the model

regressor.score(X_test, y_test)
#reinitialize X and y

X = dataset.iloc[:,0:-1].values

y = dataset.iloc[:,6:7].values
#train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#importing Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
#using the criterion of mean square error

regressor = DecisionTreeRegressor(criterion = "mse", random_state = 0)

regressor.fit(X_train,y_train)
#importing metrics to calculate the r2 score and mse

from sklearn.metrics import r2_score,mean_squared_error
#r2 score for the training set

r2_score(y_train, regressor.predict(X_train))
#mse for the training set

mean_squared_error(y_train, regressor.predict(X_train))
#r2 score for the test set

r2_score(y_test, regressor.predict(X_test))
#mse for the test set

mean_squared_error(y_test, regressor.predict(X_test))
#importing the Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
#using 100 estimators for Random Forest Calculation

#takes average of 100 predictions and trains model

regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)

regressor.fit(X_train, y_train)
#r2 score of training set

r2_score(y_train, regressor.predict(X_train))
#mse of training set

mean_squared_error(y_train, regressor.predict(X_train))
#r2 score of test set

r2_score(y_test, regressor.predict(X_test))
#mse of test set

mean_squared_error(y_test, regressor.predict(X_test))