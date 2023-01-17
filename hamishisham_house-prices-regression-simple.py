# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
correlations = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(correlations, vmax=.8, square=True);
correlations = correlations["SalePrice"].sort_values(ascending=False)
features = correlations.index[1:6]
correlations
# Checking for only missing values in Training Data
training_null = train.isnull().sum()[train.isnull().sum() != 0]
print('Missing Values in Training Data:\n')
print(training_null)
# Checking for only missing values in Testing Data
testing_null =test.isnull().sum()[test.isnull().sum() != 0]
print('Missing Values in Testing Data :\n')
print(testing_null)
null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
null
# Showing the missing values in Training Data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
# Showing the missing values in Testing Data
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #not as much missing values
null_many
null_has_meaning_features = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
for i in null_has_meaning_features:
    train[i].fillna("None", inplace=True)
    test[i].fillna("None", inplace=True)
imputer = SimpleImputer(strategy="median")
training_null = pd.isnull(train).sum()
testing_null = pd.isnull(test).sum()

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #few missing values
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
numeric_features = train.select_dtypes(include=[np.number])
cat_features = train.select_dtypes(include=[np.object])
cat_features.columns
fill_num = []

for i in numeric_features:
    if i in list(null_few.index):
        fill_num.append(i)
print(fill_num)
for i in fill_num:
    train[i].fillna(train[i].median(), inplace=True)
    test[i].fillna(test[i].median(), inplace=True)
fill_cat = []

for i in cat_features:
    if i in list(null_few.index):
        fill_cat.append(i)
print(fill_cat)
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
most_common_dictionary
counter = 0
for i in fill_cat:  
    train[i].fillna(most_common[counter], inplace=True)
    test[i].fillna(most_common[counter], inplace=True)
    counter += 1
training_null = pd.isnull(train).sum()
testing_null = pd.isnull(test).sum()

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])
null[null.sum(axis=1) > 0]
sns.distplot(np.log(train["SalePrice"]))
train["TransformedPrice"] = np.log(train["SalePrice"])
train = pd.get_dummies(train)
X_train = train.drop(["Id", "SalePrice", "TransformedPrice"], axis=1).values
y_train = train["TransformedPrice"].values
X_test = test.drop("Id", axis=1).values
X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0) #X_valid and y_valid are the validation sets
reg = linear_model.LinearRegression()
parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}
grid_reg = GridSearchCV(reg, parameters_lin, verbose=1 , scoring = "r2")
grid_reg.fit(X_training, y_training)

print("Best LinReg Model: " + str(grid_reg.best_estimator_))
print("Best Score: " + str(grid_reg.best_score_))
reg = grid_reg.best_estimator_
reg.fit(X_training, y_training)
lin_pred = reg.predict(X_valid)
r2_lin = r2_score(y_valid, lin_pred)
rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))
print("R^2 Score: " + str(r2_lin))
print("RMSE Score: " + str(rmse_lin))
scores_lin = cross_val_score(reg, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lin)))
reg.fit(X_train, y_train)
sub_predict = np.exp(reg.predict(X_valid))
sub_predict
submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": sub_predict
    })
print(submission.shape)