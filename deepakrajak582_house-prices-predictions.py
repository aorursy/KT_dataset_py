import numpy as np 
import pandas as pd 

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
#Loading the training/testing data
training = pd.read_csv("../input/train.csv")
testing = pd.read_csv("../input/test.csv")

#Lets look the rows and columns of our train datasets
training.shape
#Lets look the rows and columns of our test datasets
testing.shape
#Lets check the first few records
training.head()
#Lets see through the data
training.describe()
#Check the total number of columns
training.keys()
#creating the correlation matrix
correlations = training.corr()
#filtering the matrix only for SalesPrice and sorting descending
correlations = correlations["SalePrice"].sort_values(ascending=False)
#Lets see which features are more correlated 
correlations
#Creating a feature dataframe - first 5
features = correlations.index[1:6]
#Printing the features
features

training_null = pd.isnull(training).sum()
testing_null = pd.isnull(testing).sum()

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
null
null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #not as much missing values
null_many
#you can find these features on the description data file provided

null_has_meaning = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
                    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", 
                    "GarageCond", "PoolQC", "Fence", "MiscFeature"]
for i in null_has_meaning:
    training[i].fillna("None", inplace=True)
    testing[i].fillna("None", inplace=True)
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
training_null = pd.isnull(training).sum()
testing_null = pd.isnull(testing).sum()

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #few missing values
null_many
training.drop("LotFrontage", axis=1, inplace=True)
testing.drop("LotFrontage", axis=1, inplace=True)
null_few
training["GarageYrBlt"].fillna(training["GarageYrBlt"].median(), inplace=True)
testing["GarageYrBlt"].fillna(testing["GarageYrBlt"].median(), inplace=True)
training["MasVnrArea"].fillna(training["MasVnrArea"].median(), inplace=True)
testing["MasVnrArea"].fillna(testing["MasVnrArea"].median(), inplace=True)
training["MasVnrType"].fillna("None", inplace=True)
testing["MasVnrType"].fillna("None", inplace=True)

types_train = training.dtypes #type of each feature in data: int, float, object
num_train = types_train[(types_train != object) ] #numerical values are either type int or float
cat_train = types_train[types_train == object] #categorical values are type object

#same steps for test set also

types_test = testing.dtypes #type of each feature in data: int, float, object
num_test = types_test[(types_test != object) ] #numerical values are either type int or float
cat_test = types_test[types_test == object] #categorical values are type object

#we should convert num_train and num_test to a list to make it easier to work with
numerical_values_train = list(num_train.index)
numerical_values_test = list(num_test.index)
print(numerical_values_train)
print(numerical_values_test)
fill_num = []

for i in numerical_values_train:
    if i in list(null_few.index):
        fill_num.append(i)
print(fill_num)
for i in fill_num:
    training[i].fillna(training[i].median(), inplace=True)
    testing[i].fillna(testing[i].median(), inplace=True)
categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)
categorical_values_train
fill_cat = []

for i in categorical_values_train:
    if i in list(null_few.index):
        fill_cat.append(i)
fill_cat
def most_common_term(lst):
    lst = list(lst)
    return max(set(lst), key=lst.count)
#most_common_term finds the most common term in a series

most_common = ["Electrical", "Exterior1st", "Exterior2nd", "Functional", "KitchenQual", 
               "MSZoning", "SaleType", "Utilities", "MasVnrType"]

counter = 0
for i in fill_cat:
    most_common[counter] = most_common_term(training[i])
    counter += 1
most_common_dictionary = {fill_cat[0]: [most_common[0]], fill_cat[1]: [most_common[1]], fill_cat[2]: [most_common[2]], fill_cat[3]: [most_common[3]],
                          fill_cat[4]: [most_common[4]], fill_cat[5]: [most_common[5]], fill_cat[6]: [most_common[6]], fill_cat[7]: [most_common[7]],
                          fill_cat[8]: [most_common[8]]}
most_common_dictionary
counter = 0
for i in fill_cat:  
    training[i].fillna(most_common[counter], inplace=True)
    testing[i].fillna(most_common[counter], inplace=True)
    counter += 1
training_null = pd.isnull(training).sum()
testing_null = pd.isnull(testing).sum()

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
null[null.sum(axis=1) > 0]
sns.distplot(training["SalePrice"]);
sns.distplot(np.log(training["SalePrice"]));
training["TransformedPrice"] = np.log(training["SalePrice"])
categorical_values_train = list(cat_train.index)
categorical_values_test = list(cat_test.index)
print(categorical_values_train)
for i in categorical_values_train:
    feature_set = set(training[i])
    for j in feature_set:
        feature_list = list(feature_set)
        training.loc[training[i] == j, i] = feature_list.index(j)

for i in categorical_values_test:
    feature_set2 = set(testing[i])
    for j in feature_set2:
        feature_list2 = list(feature_set2)
        testing.loc[testing[i] == j, i] = feature_list2.index(j)

training.head()
testing.head()
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
warnings.filterwarnings("ignore", category=DeprecationWarning) 
X_train = training.drop(["Id", "SalePrice", "TransformedPrice"], axis=1).values
y_train = training["TransformedPrice"].values
X_test = testing.drop("Id", axis=1).values
from sklearn.model_selection import train_test_split #to create validation data set

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) 
#X_valid and y_valid are the validation sets
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
rf.fit(X_train, y_train)
submission_predictions = np.exp(rf.predict(X_test))
submission = pd.DataFrame({
        "Id": testing["Id"],
        "SalePrice": submission_predictions
    })

submission.to_csv("House_Prices.csv", index=False)
print(submission.shape)
