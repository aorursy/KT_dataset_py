import numpy as np 

import pandas as pd 

import seaborn as sns


import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning) 



train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

#correlations

correlations = train.corr()

correlations = correlations["SalePrice"].sort_values(ascending=False)

features = correlations.index[1:6]



#find missing data

train_null = pd.isnull(train).sum()

test_null = pd.isnull(test).sum()



#merge missing data

null = pd.concat([train_null, test_null], axis=1, keys=["Train", "Test"])



#null data multiplicity test

null_many = null[null.sum(axis=1) > 200]  #a lot of missing values

null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #not as much missing values

print(null_many)

print(null_few)



#correction of meaning mess between na and none

null_has_meaning = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]



for i in null_has_meaning:

    train[i].fillna("None", inplace=True)

    test[i].fillna("None", inplace=True)

    

#complete missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

train_null = pd.isnull(train).sum()

test_null = pd.isnull(test).sum()



#again correction of meaning mess between na and none

null = pd.concat([train_null, test_null], axis=1, keys=["Train", "Test"])

null_many = null[null.sum(axis=1) > 200]  #a lot of missing values

null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #few missing values

null_many



#data extraction

train.drop("LotFrontage", axis=1, inplace=True)

test.drop("LotFrontage", axis=1, inplace=True)

null_few



#filling missing places with median

train["GarageYrBlt"].fillna(train["GarageYrBlt"].median(), inplace=True)

test["GarageYrBlt"].fillna(test["GarageYrBlt"].median(), inplace=True)

train["MasVnrArea"].fillna(train["MasVnrArea"].median(), inplace=True)

test["MasVnrArea"].fillna(test["MasVnrArea"].median(), inplace=True)

train["MasVnrType"].fillna("None", inplace=True)

test["MasVnrType"].fillna("None", inplace=True)



#split data types integer or object for train



types_train = train.dtypes 

num_train = types_train[(types_train == int) | (types_train == float)] 

cat_train = types_train[types_train == object]



#we do the same for the test 

types_test = test.dtypes

num_test = types_test[(types_test == int) | (types_test == float)]

cat_test = types_test[types_test == object]



#we should convert num_train and num_test to a list to make it easier to work with



numerical_values_train = list(num_train.index)

numerical_values_test = list(num_test.index)



fill_num = []



for i in numerical_values_train:

    if i in list(null_few.index):

        fill_num.append(i)



for i in fill_num:

    train[i].fillna(train[i].median(), inplace=True)

    test[i].fillna(test[i].median(), inplace=True)

    

categorical_values_train = list(cat_train.index)

categorical_values_test = list(cat_test.index)





fill_cat = []



for i in categorical_values_train:

    if i in list(null_few.index):

        fill_cat.append(i)

        





def most_common_term(lst):

    lst = list(lst)

    return max(set(lst), key=lst.count)



#most_common_term finds the most common term in a series



most_common = ["Electrical", "Exterior1st", "Exterior2nd", "Functional", "KitchenQual", "MSZoning", "SaleType", "Utilities", "MasVnrType"]



counter = 0

for i in fill_cat:

    most_common[counter] = most_common_term(train[i])

    counter += 1

most_common_dictionary = {fill_cat[0]: [most_common[0]], fill_cat[1]: [most_common[1]], fill_cat[2]: [most_common[2]], fill_cat[3]: [most_common[3]],

                          fill_cat[4]: [most_common[4]], fill_cat[5]: [most_common[5]], fill_cat[6]: [most_common[6]], fill_cat[7]: [most_common[7]],

                          fill_cat[8]: [most_common[8]]}



counter = 0

for i in fill_cat:  

    train[i].fillna(most_common[counter], inplace=True)

    test[i].fillna(most_common[counter], inplace=True)

    counter += 1



train_null = pd.isnull(train).sum()

test_null = pd.isnull(test).sum()



null = pd.concat([train_null, test_null], axis=1, keys=["Training", "Testing"])

null[null.sum(axis=1) > 0]

sns.distplot(train["SalePrice"])

sns.distplot(np.log(train["SalePrice"]))

train["TransformedPrice"] = np.log(train["SalePrice"])

categorical_values_train = list(cat_train.index)

categorical_values_test = list(cat_test.index)





for i in categorical_values_train:

    feature_set = set(train[i])

    for j in feature_set:

        feature_list = list(feature_set)

        train.loc[train[i] == j, i] = feature_list.index(j)



for i in categorical_values_test:

    feature_set2 = set(test[i])

    for j in feature_set2:

        feature_list2 = list(feature_set2)

        test.loc[test[i] == j, i] = feature_list2.index(j)

        

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score 







X_train = train.drop(["Id", "SalePrice", "TransformedPrice"], axis=1).values

y_train = train["TransformedPrice"].values

X_test = test.drop("Id", axis=1).values



from sklearn.model_selection import train_test_split #to create validation data set



X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets
linreg = LinearRegression()

parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}

grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")

grid_linreg.fit(X_training, y_training)



print("Best LinReg Model: " + str(grid_linreg.best_estimator_))

print("Best Score: " + str(grid_linreg.best_score_))



linreg = grid_linreg.best_estimator_

linreg.fit(X_training, y_training)

lin_pred = linreg.predict(X_valid)

r2_lin = r2_score(y_valid, lin_pred)

rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))

print("R^2 Score: " + str(r2_lin))

print("RMSE Score: " + str(rmse_lin))



scores_lin = cross_val_score(linreg, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_lin)))
lasso = Lasso()

parameters_lasso = {"fit_intercept" : [True, False], "normalize" : [True, False], "precompute" : [True, False], "copy_X" : [True, False]}

grid_lasso = GridSearchCV(lasso, parameters_lasso, verbose=1, scoring="r2")

grid_lasso.fit(X_training, y_training)



print("Best Lasso Model: " + str(grid_lasso.best_estimator_))

print("Best Score: " + str(grid_lasso.best_score_))



lasso = grid_lasso.best_estimator_

lasso.fit(X_training, y_training)

lasso_pred = lasso.predict(X_valid)

r2_lasso = r2_score(y_valid, lasso_pred)

rmse_lasso = np.sqrt(mean_squared_error(y_valid, lasso_pred))

print("R^2 Score: " + str(r2_lasso))

print("RMSE Score: " + str(rmse_lasso))



scores_lasso = cross_val_score(lasso, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_lasso)))
ridge = Ridge()

parameters_ridge = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False], "solver" : ["auto"]}

grid_ridge = GridSearchCV(ridge, parameters_ridge, verbose=1, scoring="r2")

grid_ridge.fit(X_training, y_training)



print("Best Ridge Model: " + str(grid_ridge.best_estimator_))

print("Best Score: " + str(grid_ridge.best_score_))



ridge = grid_ridge.best_estimator_

ridge.fit(X_training, y_training)

ridge_pred = ridge.predict(X_valid)

r2_ridge = r2_score(y_valid, ridge_pred)

rmse_ridge = np.sqrt(mean_squared_error(y_valid, ridge_pred))

print("R^2 Score: " + str(r2_ridge))

print("RMSE Score: " + str(rmse_ridge))



scores_ridge = cross_val_score(ridge, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_ridge)))
dtr = DecisionTreeRegressor()

parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"], "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 5, 10], 

                  "max_features" : ["auto", "log2"]}

grid_dtr = GridSearchCV(dtr, parameters_dtr, verbose=1, scoring="r2")

grid_dtr.fit(X_training, y_training)



print("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_))

print("Best Score: " + str(grid_dtr.best_score_))



dtr = grid_dtr.best_estimator_

dtr.fit(X_training, y_training)

dtr_pred = dtr.predict(X_valid)

r2_dtr = r2_score(y_valid, dtr_pred)

rmse_dtr = np.sqrt(mean_squared_error(y_valid, dtr_pred))

print("R^2 Score: " + str(r2_dtr))

print("RMSE Score: " + str(rmse_dtr))



scores_dtr = cross_val_score(dtr, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_dtr)))
rf = RandomForestRegressor()

paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 

                 "max_features" : ["auto", "log2"]}

grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2")

grid_rf.fit(X_training, y_training)



print("Best RandomForestRegressor Model: " + str(grid_rf.best_estimator_))

print("Best Score: " + str(grid_rf.best_score_))



rf = grid_rf.best_estimator_

rf.fit(X_training, y_training)

rf_pred = rf.predict(X_valid)

r2_rf = r2_score(y_valid, rf_pred)

rmse_rf = np.sqrt(mean_squared_error(y_valid, rf_pred))

print("R^2 Score: " + str(r2_rf))

print("RMSE Score: " + str(rmse_rf))



scores_rf = cross_val_score(rf, X_training, y_training, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_rf)))
from xgboost import XGBRegressor

XGB = XGBRegressor(max_depth = 5, learning_rate = 0.05, n_estimators = 1500, reg_alpha = 0.001,

                reg_lambda = 0.000001, n_jobs = -1, min_child_weight = 3)



XGB.fit(X_train,y_train)

model_performances = pd.DataFrame({

    "Model" : ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor"],

    "Best Score" : [grid_linreg.best_score_,  grid_ridge.best_score_, grid_lasso.best_score_, grid_dtr.best_score_, grid_rf.best_score_],

    "R Squared" : [str(r2_lin)[0:5], str(r2_ridge)[0:5], str(r2_lasso)[0:5], str(r2_dtr)[0:5], str(r2_rf)[0:5]],

    "RMSE" : [str(rmse_lin)[0:8], str(rmse_ridge)[0:8], str(rmse_lasso)[0:8], str(rmse_dtr)[0:8], str(rmse_rf)[0:8]]

})

model_performances.round(4)

print("Sorted by Best Score:")

model_performances.sort_values(by="Best Score", ascending=False)



print("Sorted by R Squared:")

model_performances.sort_values(by="R Squared", ascending=False)



print("Sorted by RMSE:")

model_performances.sort_values(by="RMSE", ascending=True)



rf.fit(X_training, y_training)

submission_predictions = np.exp(rf.predict(X_test))



submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": submission_predictions

    })



submission.to_csv("prices.csv", index=False)
