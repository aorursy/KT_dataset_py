import os

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

import matplotlib

from matplotlib import pyplot as plt

%matplotlib inline



zomato = pd.read_csv("../input/zomato.csv", na_values = ["-", ""])

# Making a copy of the data to work on

data = zomato.copy()

# Renaming and removing commas in the cost column 

data = data.rename({"approx_cost(for two people)": "cost"}, axis=1)

data["cost"] = data["cost"].replace(",", "", regex = True)
# Converting numeric columns to their appropriate dtypes

data[["votes", "cost"]] = data[["votes", "cost"]].apply(pd.to_numeric)
# Group and aggregate duplicate restaurants that are listed under multiple types in listed_in(type)

grouped = data.groupby(["name", "address"]).agg({"listed_in(type)" : list})

newdata = pd.merge(grouped, data, on = (["name", "address"]))
# Drop rows which have duplicate information in "name", "address" and "listed_in(type)_x"

newdata["listed_in(type)_x"] = newdata["listed_in(type)_x"].astype(str) # converting unhashable list to a hashable type

newdata.drop_duplicates(subset = ["name", "address", "listed_in(type)_x"], inplace = True)

# Converting the restaurant names to rownames 

newdata.index = newdata["name"]
# Dropping unnecessary columns

newdata.drop(["name", "url", "phone", "listed_in(city)", "listed_in(type)_x", "address", "dish_liked",  "listed_in(type)_y", "menu_item", "cuisines", "reviews_list"], axis = 1, inplace = True)
# Transforming the target (restaurant ratings)



# Extracting the first three characters of each string in "rate"

newdata["rating"] = newdata["rate"].str[:3] 

# Removing rows with "NEW" in ratings as it is not a predictable level

newdata = newdata[newdata.rating != "NEW"] 

# Dropping rows that have missing values in ratings 

newdata = newdata.dropna(subset = ["rating"])

# Converting ratings to a numeric column so we can discretize it

newdata["rating"] = pd.to_numeric(newdata["rating"])
# Discretizing the ratings into a categorical feature with 4 classes

newdata["rating"] = pd.cut(newdata["rating"], bins = [0, 3.0, 3.5, 4.0, 5.0], labels = ["0", "1", "2", "3"])
# Checking the number of restaurants in each rating class

np.unique(newdata["rating"], return_counts = True)
# Dropping the original rating column

newdata.drop("rate", axis = 1, inplace = True)
newdata.describe(include = "all")
# Separating the predictors and target

predictors = newdata.drop("rating", axis = 1)

target = newdata["rating"]
# Splitting the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(predictors, target, random_state = 0, test_size = 0.3)
# Preprocessing the predictors

num_cols = ["votes", "cost"]

cat_cols = ["location", "rest_type", "online_order", "book_table"]



num_imputer = SimpleImputer(strategy = "median")

# Imputing numeric columns with the median (not mean because of the high variance)

num_imputed = num_imputer.fit_transform(X_train[num_cols])

scaler = StandardScaler()

# Scaling the numeric columns to have a mean of 0 and standard deviation of 1

num_preprocessed = pd.DataFrame(scaler.fit_transform(num_imputed), columns = num_cols)



cat_imputer = SimpleImputer( strategy = "most_frequent")

# Imputing categorical columns with the mode

cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_cols]), columns = cat_cols)

# Dummifying the categorical columns

cat_preprocessed = pd.DataFrame(pd.get_dummies(cat_imputed, prefix = cat_cols, drop_first = True))



train_predictors = pd.concat([num_preprocessed, cat_preprocessed], axis=1)
test_num_imputed = num_imputer.transform(X_test[num_cols])

test_num_preprocessed = pd.DataFrame(scaler.transform(test_num_imputed), columns = num_cols)



test_cat_imputed = pd.DataFrame(cat_imputer.transform(X_test[cat_cols]), columns = cat_cols)

test_cat_preprocessed = pd.DataFrame(pd.get_dummies(test_cat_imputed, prefix = cat_cols, drop_first = True))

                                    

test_predictors = pd.concat([test_num_preprocessed, test_cat_preprocessed], axis=1)



# Accounting for missing columns in the test set caused by dummification

missing_cols = set(train_predictors) - set(test_predictors)

# Add missing columns to test set with default value equal to 0

for c in missing_cols:

    test_predictors[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

test_predictors = test_predictors[train_predictors.columns]
dt = DecisionTreeClassifier()

dt.fit(train_predictors, y_train)

pred_train = dt.predict(train_predictors)

pred_test = dt.predict(test_predictors)
accuracy_score(y_train, pred_train)
accuracy_score(y_test, pred_test)
rf = RandomForestClassifier(criterion = "gini", n_estimators = 250, max_depth = 10, 

                            max_features = 50, min_samples_split = 4, random_state = 0)

rf.fit(train_predictors, y_train)

pred_train = rf.predict(train_predictors)

pred_test = rf.predict(test_predictors)
accuracy_score(y_train, pred_train)
accuracy_score(y_test, pred_test)
cohen_kappa_score(y_train, pred_train)
cohen_kappa_score(y_test, pred_test)
print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))
# Inspecting class counts in the train predictions

np.unique(pred_train, return_counts = True)
# Doing the same for the test predictions

np.unique(pred_test, return_counts = True)
rf = RandomForestClassifier(criterion = "gini", n_estimators = 250, max_depth = 10, 

                            max_features = 50, min_samples_split = 4, random_state = 0,

                           class_weight = "balanced")

rf.fit(train_predictors, y_train)

pred_train = rf.predict(train_predictors)

pred_test = rf.predict(test_predictors)
# Inspecting class counts in the train predictions

np.unique(pred_train, return_counts = True)
# Doing the same for the test predictions

np.unique(pred_test, return_counts = True)
np.unique(y_train, return_counts = True)
np.unique(y_test, return_counts = True)
# Building an XGBoost classifier

xgb = XGBClassifier(n_estimators = 250, max_depth = 20, gamma = 2, learning_rate = 0.001, random_state = 0)



xgb.fit(train_predictors, y_train)

pred_train = xgb.predict(train_predictors)

pred_test = xgb.predict(test_predictors)
accuracy_score(y_train, pred_train)
accuracy_score(y_test, pred_test)
cohen_kappa_score(y_train, pred_train)
cohen_kappa_score(y_test, pred_test)
print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))
# Inspecting class counts in the train predictions

np.unique(pred_train, return_counts = True)
# Doing the same for the test predictions

np.unique(pred_test, return_counts = True)
# Visualizing a feature importances plot



plt.figure(figsize = (20, 10))

feat_importances = pd.Series(xgb.feature_importances_, index=train_predictors.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()