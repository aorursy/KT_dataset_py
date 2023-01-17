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
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import os
print(os.listdir("../input"))

housing_data = pd.read_csv('../input/housing.csv',sep=",")
housing_data.head()
housing_data.shape
housing_data.isnull().any()
housing_data.describe()
# Applying mean normalization on features
def meannormalize(x):
    return (x - x.mean())/x.std()

housing_data = housing_data.apply(meannormalize)
housing_data.describe()
corr_df = housing_data.corr()

print(corr_df)
plt.matshow(corr_df)
#plt.scatter(housing_data['LSTAT'],housing_data['MEDV'],c="b")
#plt.show()
plt.figure(figsize=(20, 5))
features = ['LSTAT','PTRATIO','RM']
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = housing_data[col]
    y = housing_data['MEDV']
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
X = housing_data[['LSTAT','RM','PTRATIO']]
y = housing_data['MEDV'].values

#X = X.reshape(-1,1)
y = y.reshape(-1,1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)
# Make predictions using the training set first
df_y_train_pred = regr.predict(X_train)

# The mean squared error, lower the value better it is. 0 means perfect prediction
print("Mean squared error of training set: %.2f"%
      mean_squared_error(y_train, df_y_train_pred))

# Explained variance score: 1 is perfect prediction
print('R2 variance score of training set: %.2f' % r2_score(y_train, df_y_train_pred))



# Make predictions using the testing set
df_y_pred = regr.predict(X_test)

# print('Coefficients: \n', regr.coef_)
# print('Intercept: \n', regr.intercept_)

# The mean squared error, lower the value better it is. 0 means perfect prediction
print("Mean squared error of testing set: %.2f"%
      mean_squared_error(y_test, df_y_pred))

# Explained variance score: 1 is perfect prediction
print('R2 Variance score of testing set: %.2f' % r2_score(y_test, df_y_pred))

#calculating adjusted r2
N = y_test.size
p = X_train.shape[1]
adjr2score = 1 - ((1-r2_score(y_test, df_y_pred))*(N - 1))/ (N - p - 1)
print("Adjusted R^2 Score %.2f" % adjr2score)

# Plot outputs
#ax = plt.scatter(X_test['RM'], y_test,color='DarkBlue', label='Group 1');

#plt.scatter(X_test['LSTAT'], y_test, color='DarkGreen', label='Group 2');

# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, df_y_pred, color='red', linewidth=1)

# plt.xticks(())
# plt.yticks(())

# plt.show()
from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_regression_model(degree):
    #"Creates a polynomial regression model for the given degree"
    poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)
    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    
    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
    
    # evaluating the model on training dataset
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)
    
    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)
    
    print("\n")
    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))
    
    print("\n")
    
    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
    print("\n")
create_polynomial_regression_model(3)