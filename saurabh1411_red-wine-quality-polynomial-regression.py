'''

# In the previous kernal for the same data-set "winequality-red.csv", the mean-squared error of the

# predictor is 0.475(approx), which was performed in Linear Regression. To reduce the mean error the

# model was predicted usinh polynomial regression and the error was reduced.



# Import the following packages pandas, numpy and sklearn.

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt



# Load the data-set.

df = pd.read_csv('../input/winequality-red.csv')



# Quality the parameter to be predicted is represented as X.

X = df[['quality']]

# All the input parameters used to predict the value are represented as y.

y = df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]



# Data-set is divided into test data and train data based on test_size variable.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Fit the training data into polynomial model of degree 4 (appropriate value as per the data-set).

model = PolynomialFeatures(degree= 4)

y_ = model.fit_transform(y)

y_test_ = model.fit_transform(y_test)



# Fit and predict the obtained polynomial model into linear regression.

# To Understand the relation between linear and polynomial regression visit the below link.

# http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions

lg = LinearRegression()

lg.fit(y_,X)

predicted_data = lg.predict(y_test_)

predicted_data = np.round_(predicted_data)



# Display the mean squared error between the predicted data and test data.

print (mean_squared_error(X_test,predicted_data))

# Display the values of predicted_data

print (predicted_data)



# As seen the mean_squared_error is reduced to 0.05 far less than that of linear regressions 0.475.'''
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
df = pd.read_csv('../input/winequality-red.csv')



# Quality the parameter to be predicted is represented as X.

X = df[['quality']]

# All the input parameters used to predict the value are represented as y.

y = df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]

# Data-set is divided into test data and train data based on test_size variable.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the training data into polynomial model of degree 4 (appropriate value as per the data-set).

model = PolynomialFeatures(degree= 4)

y_ = model.fit_transform(y)

y_test_ = model.fit_transform(y_test)
y_.shape
# Fit and predict the obtained polynomial model into linear regression.

# To Understand the relation between linear and polynomial regression visit the below link.

# http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions

lg = LinearRegression()

lg.fit(y_,X)

predicted_data = lg.predict(y_test_)

predicted_data = np.round_(predicted_data)

# Display the mean squared error between the predicted data and test data.

print (mean_squared_error(X_test,predicted_data))

# Display the values of predicted_data

print (predicted_data)



# As seen the mean_squared_error is reduced to 0.05 far less than that of linear regressions 0.475.