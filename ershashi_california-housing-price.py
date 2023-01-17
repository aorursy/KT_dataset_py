#Importing Libraries
import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
#Problem statement 1 : Load Data
#Reading data file
calf = pd.read_csv('../input/housing.csv')
#Problem statement 2 : Handel Missing data
#Filling missing values with Mean_value
calf.fillna(calf.mean(), inplace=True)
calf.head(2)
#Problem statement 3 : Categorical data encoding
#Select Columns to encode
columnsToEncode=calf.select_dtypes(include=[object]).columns

#Encode and drop first column
calf = pd.get_dummies(calf, columns=columnsToEncode, drop_first=True)
calf.head(3)

#Creating arrays for features and response variable
X = calf.drop(['longitude','latitude','housing_median_age','median_house_value'], axis=1).values
y = calf['median_house_value'].values
#Problem statement 4 : spliting dataset
#spliting training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#Problem statement 5 : standardize data
#Intantitiate
std_scale = preprocessing.StandardScaler()

#Fit & transform training features
scaled_X_train = std_scale.fit_transform(X_train)

#Fit & transform test features
scaled_X_test = std_scale.fit_transform(X_test)
scaled_X_train[1,:]
scaled_X_test[1,:]
#Problem statement 6 : Perform Linear Regression
# Creating linear regression model
lr = linear_model.LinearRegression()
lr.fit(scaled_X_train, y_train)

#Predictout for test dataset
y_predict = lr.predict(scaled_X_test)
#Linear Regression score
lr.score(scaled_X_test, y_test)
# Compute and print R^2 and RMSE

print("R^2: {}".format(lr.score(scaled_X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test,y_predict))

print("Root Mean Squared Error: {}".format(rmse))
import statsmodels.formula.api as smf
lm = smf.ols(formula='median_house_value ~ total_bedrooms', data=calf).fit()
print(lm.summary())
lm.conf_int()
calf.columns
#Problem statement 7 : Decission Tree
clf=DecisionTreeClassifier(max_leaf_nodes=25)
clf=clf.fit(scaled_X_train, y_train)

predictions = clf.predict(scaled_X_test)
predictions
#Problem Statement 7 : Decission tree prediction accuracy
#accuracy_score(y_test, predictions)
#Problem statement 9 : perform Linear Regression with one independent variable
#Extracting median_income feature value from training features set
median_income_X_train = X_train[:,4:5]
median_income_X_test = X_test[:,4:5]

#creating linear regression model for one independent variable
lr_mi = linear_model.LinearRegression()
lr_mi.fit(median_income_X_train, y_train)

y_predict_mi = lr_mi.predict(median_income_X_test)
lr_mi.score(median_income_X_test, y_test)
y_test = y_test.reshape(-1,1)
#Problem statement 9 : Plot for test data Vs regression line
plt.scatter(median_income_X_test, y_test/10000, color='blue', s=10)
plt.plot(median_income_X_test, y_predict_mi/10000, color='black', linewidth=3)
plt.ylabel('Median house value /10000 ($) -->')
plt.xlabel('Median Income -->')
plt.show()


