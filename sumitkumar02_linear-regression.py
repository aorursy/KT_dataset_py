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
bottle = pd.read_csv('../input/bottle.csv')
bottle.head(5)
bottle.shape
bottle.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



correlation_matrix = bottle.corr().round(2)

# annot = True to print the values inside the square

sns.heatmap(data=correlation_matrix, annot=True)
bottle = bottle[['Salnty', 'T_degC']]

bottle.columns = ['Sal', 'Temp']

bottle = bottle[:][:500]
print(bottle.head(10))
correlation_matrix = bottle.corr().round(2)

# annot = True to print the values inside the square

sns.heatmap(data=correlation_matrix, annot=True)
plt.figure(figsize=(20, 5))



features = ['Sal']

target = bottle['Temp']



for i, col in enumerate(features):

    plt.subplot(1, len(features) , i+1)

    x = bottle[col]

    y = target

    plt.scatter(x, y, marker='o')

    plt.title(col)

    plt.xlabel(col)

    plt.ylabel('Temp')
bottle.fillna(method='ffill', inplace=True)
X = pd.DataFrame(np.c_[bottle['Sal']], columns = ['Sal'])

Y = bottle['Temp']
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



lin_model = LinearRegression()

lin_model.fit(X_train, Y_train)
accuracy = lin_model.score(X_test, Y_test)

print(accuracy)
from sklearn.metrics import r2_score

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

rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

r2 = r2_score(Y_test, y_test_predict)



print("The model performance for testing set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))
plt.scatter(X_test, Y_test, color='b')

plt.plot(X_test,y_test_predict, color='k')

plt.show()
X_test.shape
Y_test.shape
X_train.shape
Y_train.shape
from sklearn.preprocessing import PolynomialFeatures





poly_features = PolynomialFeatures(degree=2)

  

  # transforms the existing features to higher degree features.

X_train_poly = poly_features.fit_transform(X_train)

X_test_poly = poly_features.fit_transform(X_test)



  

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

    

X_train_poly.shape
X_test_poly.shape
Y_test.shape