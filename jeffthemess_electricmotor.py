# Import libraries

import numpy as np

import pandas as pd

from pandas import Series, DataFrame

from sklearn import linear_model

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import statsmodels.api as sm

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

%matplotlib inline
# import the data set

motor_data = pd.read_csv("../input/pmsm_temperature_data.csv")

motor_data.head()
# Display the shape

motor_data.shape
# Display the data types

motor_data.dtypes
# Display any missing values from the data

motor_data.isnull().sum()
# Display the statisitcsal desctiption

motor_data.describe().T
# Display any correlation

motor_corr = motor_data.corr()

motor_corr
# Display the correlation data with non-numerical values

for col in list(motor_data.columns):

    motor_corr[col] = pd.cut(motor_corr[col],

                            (-1, -0.5, -.1, 0.1, 0.5, 1),

                            labels=["NegStrong", "NegMedium", "Weak",

                                   "PosMedium", "PosStrong"])

motor_corr
# Save the response variable

Y = motor_data['pm']
# Save the predictor variables

X = motor_data.drop('pm', axis=1)
# Standardize the predictor variables

sc = StandardScaler()

sc.fit(X)

X_std = sc.transform(X)
# Feature Selection with backward elimiation use using statsmodel linear regression

# add the Y-intercept

X_std0 = sm.add_constant(X_std)
# Run OLS on the data

Y_model0 = sm.OLS(Y, X_std0)

Y_model0 = Y_model0.fit()

Y_model0 = Y_model0.summary()

Y_model0
# Use Linear Regression with cross_val_score

lr = linear_model.LinearRegression()

scores = cross_val_score(lr, X_std, Y, cv=10)

print("Mean R-Squared Score", sum(scores)/len(scores))
# Use train_tet_split on the data

X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y,

                                                   random_state=33,

                                                   test_size=0.33)
# Train the data

lr = linear_model.LinearRegression()

model = lr.fit(X_train, Y_train)
# Predict on the test data

predictions = lr.predict(X_test)

predictions[:5]
# Display the score

print("R-Sqaured Score:", model.score(X_test, Y_test))
# Plot the scores

plt.scatter(predictions, Y_test)

plt.xlabel("Predicted values")

plt.ylabel("Actual Values");