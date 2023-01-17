# Import necessary libraies

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Load training and test data

train_df_org = pd.read_csv("../input/train.csv", header=0)

test_df_org = pd.read_csv("../input/test.csv", header=0)
all_df = pd.concat((train_df_org.loc[:,'MSSubClass':'SaleCondition'],

                      test_df_org.loc[:,'MSSubClass':'SaleCondition']))



all_df = pd.get_dummies(all_df)

all_df = all_df.fillna(all_df.mean())



train_df = all_df[:train_df_org.shape[0]]

test_df = all_df[train_df_org.shape[0]:]



# Generation data

train_data_x = train_df.values

train_data_y = train_df_org['SalePrice'].values

test_data_x  = test_df.values
# Import necessary library

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import RANSACRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import learning_curve

from sklearn.cross_validation import train_test_split

from sklearn import preprocessing



# Create regressors

reglist = []

reglist.append((LinearRegression(), "Linear Regression"))

reglist.append((Ridge(), "Ridge Regression"))

reglist.append((Lasso(), "Lasso Regression"))

reglist.append((RANSACRegressor(), "RANSAC, Regressor"))

reglist.append((RandomForestRegressor(), "Random Forest Regressor"))

reglist.append((GradientBoostingRegressor(), "Gradient Boosting Regressor"))

reglist.append((DecisionTreeRegressor(), "Decision Tree Regressor"))

reglist.append((SVR(), "SVM Regression"))

reglist.append((MLPRegressor(), "Neural Network regression"))
# Divide training set and test set.

X_train, X_test, y_train, y_test = train_test_split(train_data_x, train_data_y, test_size = 0.3, random_state=0)



# Fit to training data and calculate score by test set.

pred = []

index = 0

for reg in reglist:

    reg[0].fit(X_train, y_train)

    print("#{0} : {1} : {2}".format(index, reg[1], reg[0].score(X_test, y_test)))

    pred.append(reg[0].predict(X_test))

    index += 1
# Output predict by test data

result_df = pd.DataFrame()

result_df['Expected'] = y_test

result_df['Actual1'] = pred[1]

result_df['Actual2'] = pred[4]

result_df['Actual3'] = pred[5]

result_df['Actual4'] = pred[6]

result_df['Actual5'] = pred[8]

result_df['Error1'] = result_df['Expected'] - result_df['Actual1']

result_df['Error2'] = result_df['Expected'] - result_df['Actual2']

result_df['Error3'] = result_df['Expected'] - result_df['Actual3']

result_df['Error4'] = result_df['Expected'] - result_df['Actual4']

result_df['Error5'] = result_df['Expected'] - result_df['Actual5']

result_df.describe()
# Fit to whole training dataset

reglist[5][0].fit(train_data_x, train_data_y)



# Predict by test dataset

last_pred = reglist[5][0].predict(test_data_x)



# Create submit data

submit_df = test_df_org.copy()

submit_df["SalePrice"] = last_pred

submit_df[["Id", "SalePrice"]].to_csv("predict.csv", index=False)