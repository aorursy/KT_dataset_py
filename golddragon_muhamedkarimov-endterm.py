# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from imblearn.over_sampling import SMOTE

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats.stats import pearsonr

import xgboost as xgb



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()
test.head()
print("Train:", train.shape, "\nTest: ", test.shape)
correlations = train.corr()

correlations = correlations["SalePrice"].sort_values(ascending=False)

features = correlations.index[1:6]

print(correlations)
train_null = pd.isnull(train).sum()

test_null = pd.isnull(test).sum()



null = pd.concat([train_null, test_null], axis=1, keys=["Train", "Test"])
null_many = null[null.sum(axis=1) > 200]

null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]

null_many
nullm = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", 

         "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

for i in nullm:

    train[i].fillna("None", inplace=True)

    test[i].fillna("None", inplace=True)
train_null = pd.isnull(train).sum()

test_null = pd.isnull(test).sum()



null = pd.concat([train_null, test_null], axis=1, keys=["Train", "Test"])
null_many = null[null.sum(axis=1) > 200]

null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]

null_many
train.drop("LotFrontage", axis=1, inplace=True)

test.drop("LotFrontage", axis=1, inplace=True)

null_few
train["GarageYrBlt"].fillna(train["GarageYrBlt"].median(), inplace=True)

test["GarageYrBlt"].fillna(test["GarageYrBlt"].median(), inplace=True)

train["MasVnrArea"].fillna(train["MasVnrArea"].median(), inplace=True)

test["MasVnrArea"].fillna(test["MasVnrArea"].median(), inplace=True)

train["MasVnrType"].fillna("None", inplace=True)

test["MasVnrType"].fillna("None", inplace=True)
types_train = train.dtypes

num_train = types_train[(types_train == int) | (types_train == float)]

cat_train = types_train[types_train == object]



types_test = test.dtypes

num_test = types_test[(types_test == int) | (types_test == float)]

cat_test = types_test[types_test == object]
numval_train = list(num_train.index)

numval_test = list(num_test.index)

print(*numval_train, sep=', ')
fill_num = []

for i in numval_train:

    if i in list(null_few.index):

        fill_num.append(i)

print(*fill_num, sep=', ')
for i in fill_num:

    train[i].fillna(train[i].median(), inplace=True)

    test[i].fillna(test[i].median(), inplace=True)
catval_train = list(cat_train.index)

catval_test = list(cat_test.index)

print(*catval_train, sep=', ')
fill_cat = []

for i in catval_train:

    if i in list(null_few.index):

        fill_cat.append(i)

print(*fill_cat, sep=', ')
most_common = ["Electrical", "Exterior1st", "Exterior2nd", "Functional", "KitchenQual", "MSZoning", "SaleType", "Utilities", "MasVnrType"]



counter = 0

for i in fill_cat:

    a = list(train[i])

    most_common[counter] = max(set(a), key=a.count)

    counter += 1
most_common_dict = {}

for i in range(len(most_common)):

    most_common_dict[fill_cat[i]] = [most_common[i]]

most_common_dict
counter = 0

for i in fill_cat:  

    train[i].fillna(most_common[counter], inplace=True)

    test[i].fillna(most_common[counter], inplace=True)

    counter += 1
train_null = pd.isnull(train).sum()

test_null = pd.isnull(test).sum()



null = pd.concat([train_null, test_null], axis=1, keys=["Train", "Test"])

null[null.sum(axis=1) > 0]
sns.distplot(train["SalePrice"])
sns.distplot(np.log(train["SalePrice"]))
train["TransformedPrice"] = np.log(train["SalePrice"])



catval_train = list(cat_train.index)

catval_test = list(cat_test.index)

print(*catval_train, sep=', ')
for i in catval_train:

    feature_set = set(train[i])

    for j in feature_set:

        feature_list = list(feature_set)

        train.loc[train[i] == j, i] = feature_list.index(j)



for i in catval_test:

    feature_set2 = set(test[i])

    for j in feature_set2:

        feature_list2 = list(feature_set2)

        test.loc[test[i] == j, i] = feature_list2.index(j)
train.head()
test.head()
x_train = train.drop(["Id", "SalePrice", "TransformedPrice"], axis=1).values

y_train = train["TransformedPrice"].values

x_test2 = test.drop("Id", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
linreg = LinearRegression()

parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}

grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")

grid_linreg.fit(x_train, y_train)



print("Best LinReg Model: " + str(grid_linreg.best_estimator_))

print("Best Score: " + str(grid_linreg.best_score_))
linreg = grid_linreg.best_estimator_

linreg.fit(x_train, y_train)

lin_pred = linreg.predict(x_test)

r2_lin = r2_score(y_test, lin_pred)

rmse_lin = np.sqrt(mean_squared_error(y_test, lin_pred))

print("R^2 Score: " + str(r2_lin))

print("RMSE Score: " + str(rmse_lin))
dtr = DecisionTreeRegressor()

parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"], "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 5, 10], 

                  "max_features" : ["auto", "log2"]}

grid_dtr = GridSearchCV(dtr, parameters_dtr, verbose=1, scoring="r2")

grid_dtr.fit(x_train, y_train)



print("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_))

print("Best Score: " + str(grid_dtr.best_score_))
dtr = grid_dtr.best_estimator_

dtr.fit(x_train, y_train)

dtr_pred = dtr.predict(x_test)

r2_dtr = r2_score(y_test, dtr_pred)

rmse_dtr = np.sqrt(mean_squared_error(y_test, dtr_pred))

print("R^2 Score: " + str(r2_dtr))

print("RMSE Score: " + str(rmse_dtr))
rf = RandomForestRegressor()

paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 

                 "max_features" : ["auto", "log2"]}

grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2")

grid_rf.fit(x_train, y_train)



print("Best RandomForestRegressor Model: " + str(grid_rf.best_estimator_))

print("Best Score: " + str(grid_rf.best_score_))
rf = grid_rf.best_estimator_

rf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)

r2_rf = r2_score(y_test, rf_pred)

rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))

print("R^2 Score: " + str(r2_rf))

print("RMSE Score: " + str(rmse_rf))
rf.fit(x_train, y_train)
submission_predictions = np.exp(rf.predict(x_test2))
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": submission_predictions

    })



submission.to_csv("submission.csv", index=False)

print(submission.shape)