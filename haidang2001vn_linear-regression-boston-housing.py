import numpy as np

import matplotlib.pyplot as plt 



import pandas as pd  

import seaborn as sns 



%matplotlib inline
from sklearn.datasets import load_boston



boston_dataset = load_boston()



# boston_dataset is a dictionary

# let's check what it contains

boston_dataset.keys()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston.head()
boston['MEDV'] = boston_dataset.target
# check for missing values in all the columns

boston.isnull().sum()
# set the size of the figure

sns.set(rc={'figure.figsize':(11.7,8.27)})



# plot a histogram showing the distribution of the target values

sns.distplot(boston['MEDV'], bins=30)

plt.show()
# compute the pair wise correlation for all columns  

correlation_matrix = boston.corr().round(2)
# use the heatmap function from seaborn to plot the correlation matrix

# annot = True to print the values inside the square

sns.heatmap(data=correlation_matrix, annot=True)
plt.figure(figsize=(20, 5))



features = ['LSTAT', 'RM']

target = boston['MEDV']



for i, col in enumerate(features):

    plt.subplot(1, len(features) , i+1)

    x = boston[col]

    y = target

    plt.scatter(x, y, marker='o')

    plt.title(col)

    plt.xlabel(col)

    plt.ylabel('MEDV')
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])

Y = boston['MEDV']
from sklearn.model_selection import train_test_split



# splits the training and test data set in 80% : 20%

# assign random_state to any value.This ensures consistency.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



lin_model = LinearRegression()

lin_model.fit(X_train, Y_train)
# model evaluation for training set



y_train_predict = lin_model.predict(X_train)

rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))

r2 = r2_score(Y_train, y_train_predict)



print("The model performance for training set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))

print("\n")



# model evaluation for testing set



y_test_predict = lin_model.predict(X_test)

# root mean square error of the model

rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))



# r-squared score of the model

r2 = r2_score(Y_test, y_test_predict)



print("The model performance for testing set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))
# plotting the y_test vs y_pred

# ideally should have been a straight line

plt.scatter(Y_test, y_test_predict)

plt.show()
from sklearn.preprocessing import PolynomialFeatures



def create_polynomial_regression_model(degree):

  "Creates a polynomial regression model for the given degree"

  poly_features = PolynomialFeatures(degree=degree)

  

  # transform the features to higher degree features.

  X_train_poly = poly_features.fit_transform(X_train)

  

  # fit the transformed features to Linear Regression

  poly_model = LinearRegression()

  poly_model.fit(X_train_poly, Y_train)

  

  # predicting on training data-set

  y_train_predicted = poly_model.predict(X_train_poly)

  

  # predicting on test data-set

  y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

  

  # evaluating the model on training dataset

  rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))

  r2_train = r2_score(Y_train, y_train_predicted)

  

  # evaluating the model on test dataset

  rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))

  r2_test = r2_score(Y_test, y_test_predict)

  

  print("The model performance for the training set")

  print("-------------------------------------------")

  print("RMSE of training set is {}".format(rmse_train))

  print("R2 score of training set is {}".format(r2_train))

  

  print("\n")

  

  print("The model performance for the test set")

  print("-------------------------------------------")

  print("RMSE of test set is {}".format(rmse_test))

  print("R2 score of test set is {}".format(r2_test))
create_polynomial_regression_model(2)