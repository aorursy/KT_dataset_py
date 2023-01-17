pwd
%matplotlib inline
import matplotlib.pyplot as plt
import sklearn.preprocessing

import numpy as np
import pandas as pd
import sys

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from pandas import DataFrame, read_csv
from sklearn import preprocessing
pd.set_option('display.max_columns', 500)
car_data=pd.read_csv("../input/Automobile.csv")
car_data.head()
from sklearn.preprocessing import LabelEncoder

list_of_feaatures_to_encode = ['make','fuel_type','aspiration','number_of_doors','body_style','drive_wheels','engine_location','engine_type','number_of_cylinders','fuel_system']
le = LabelEncoder()

for i in list_of_feaatures_to_encode:
    enc = le.fit(np.unique(car_data[i].values))
    print(enc.classes_)
    car_data[i] = le.fit_transform(car_data[i])
for i in list_of_feaatures_to_encode:
    enc = le.fit(np.unique(car_data[i].values))
    print(enc.classes_)
    car_data[i] = le.fit_transform(car_data[i])
sns.pairplot(car_data, x_vars=['symboling','normalized_losses','make','curb_weight','engine_size','compression_ratio','horsepower','peak_rpm','city_mpg'], y_vars='price', size=7, aspect=0.7, kind='reg')
feature_cols = ['normalized_losses','number_of_doors','wheel_base','number_of_cylinders','engine_size','horsepower','peak_rpm']
X = car_data[feature_cols]
print(type(X))
print(X.shape)

Y = car_data['price']
print(type(Y))
print(Y.shape)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
reg = LinearRegression()
reg.fit(X_train,Y_train)
print(reg.intercept_)
print(reg.coef_)
list(zip(feature_cols, reg.coef_))
Y_pred = reg.predict(X_test)
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
feature_cols = ['horsepower','make','wheel_base','number_of_cylinders','fuel_type','aspiration','drive_wheels','engine_location','engine_type','length','width']
#'wheel_base','number_of_cylinders','fuel_type','aspiration','drive_wheels','engine_location','engine_type','length','width'
# use the list to select a subset of the original DataFrame
X = car_data[feature_cols]

# select a Series from the DataFrame
Y = car_data.price

# split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
print(reg.intercept_)
print(reg.coef_)

# fit the model to the training data (learn the coefficients)
reg.fit(X_train, Y_train)

# make predictions on the testing set
Y_pred = reg.predict(X_test)

# compute the RMSE of our predictions

print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
reg.predict([[111,0,88.6,0,0,0,0,0,0,168.8,64.0]])