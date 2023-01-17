import numpy as np

from sklearn import datasets



# Load the diabetes dataset

diabetes = datasets.load_diabetes()





# Use only one feature

diabetes_X = diabetes.data[:, np.newaxis, 2]



# Split the data into training/testing sets

diabetes_X_train = diabetes_X[:-20]

diabetes_X_test = diabetes_X[-20:]



# Split the targets into training/testing sets

diabetes_y_train = diabetes.target[:-20]

diabetes_y_test = diabetes.target[-20:]
import matplotlib.pyplot as plt

import numpy as np

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(diabetes_X_train, diabetes_y_train)



# Make predictions using the testing set

diabetes_y_pred = regr.predict(diabetes_X_test)



# The coefficients

print('Coefficients: \n', regr.coef_)

# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))



# Plot outputs

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')

plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)



plt.show()
import matplotlib.pyplot as plt

import numpy as np

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score



# Create linear regression object

regr = linear_model.Ridge(alpha=0.3)



# Train the model using the training sets

regr.fit(diabetes_X_train, diabetes_y_train) 



# Make predictions using the testing set

diabetes_y_pred = regr.predict(diabetes_X_test)



# The coefficients

print('Coefficients: \n', regr.coef_)

# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))



# Plot outputs

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')

plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)



plt.show()



import matplotlib.pyplot as plt

import numpy as np

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score



# Create linear regression object

regr = linear_model.Lasso(alpha=0.1)



# Train the model using the training sets

regr.fit(diabetes_X_train, diabetes_y_train) 



# Make predictions using the testing set

diabetes_y_pred = regr.predict(diabetes_X_test)



# The coefficients

print('Coefficients: \n', regr.coef_)

# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))



# Plot outputs

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')

plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)



plt.show()
from sklearn.preprocessing import PolynomialFeatures



X = np.arange(6).reshape(3, 2)

poly = PolynomialFeatures(2)

print(poly.fit_transform(X))

poly = PolynomialFeatures(interaction_only=True)

# If true, only interaction features are produced: features that are products of at most degree distinct input features

print(poly.fit_transform(X))
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt



polynomial_features = PolynomialFeatures(degree=3)



diabetes_X_poly_train = polynomial_features.fit_transform(diabetes_X_train)

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_poly_train, diabetes_y_train)

diabetes_y_pred = regr.predict(polynomial_features.fit_transform(diabetes_X_test))





# The coefficients

print('Coefficients: \n', regr.coef_)

# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))



# Plot outputs

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')

plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)



plt.show()
from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)

clf.predict(X[:2, :])

print(clf.predict_proba(X[:2, :]))

print(clf.score(X, y))