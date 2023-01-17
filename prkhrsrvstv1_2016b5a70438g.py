# Imports

import csv

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
# Read the training data file

train_data = pd.read_csv("../input/eval-lab-1-f464-v2/train.csv")

# Basic info about the dataframe

train_data.info()
# Shuffle

train_data = train_data.sample(frac=1.0)

# Separate out the features and target columns

X_train, X_test, y_train, y_test = train_test_split(train_data.loc[:, "feature1":"feature11"], train_data.loc[:, "rating"], test_size=0.000008)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Convert "type" column to 0-1 int

X_train["type"] = [1 if x == "new" else 0 for x in X_train["type"]]

X_test["type"] = [1 if x == "new" else 0 for x in X_test["type"]]

print(train_data["type"])

print(X_train["type"])
# Deal with missing values by imputation

print(X_train.isna().values.any())

X_train = IterativeImputer().fit_transform(X_train)

X_test = IterativeImputer().fit_transform(X_test)

print(pd.DataFrame(X_train).isna().values.any())
# Define and fit a model

from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier)



# 1. This one got 0.64291 public, and 0.64688 private

model = ExtraTreesRegressor(n_estimators=800).fit(X_train, y_train)



# 2. This one got 0.64051 public, and 0.65476 private

# model = ExtraTreesRegressor(n_estimators=400).fit(X_train, y_train)
# Read the testing data file

test_data = pd.read_csv("../input/eval-lab-1-f464-v2/test.csv")

# Modify the "type" column

test_data["type"] = [1 if x == "new" else 0 for x in test_data["type"]]

# Imputation on test data

test_data = IterativeImputer().fit_transform(test_data)

# Separate out the id column

test_id, test_data = test_data[:, 0], test_data[:, 1:]
# Make predictions on the test data

predictions = model.predict(test_data)
# Write the predictions to a csv file

submission = open("submission.csv", "w")

writer = csv.writer(submission)

writer.writerow(["id", "rating"])

for i in range(len(predictions)):

    writer.writerow([np.int32(test_id[i]), np.round(predictions[i])])

submission.close()
print(model.score(X_test, y_test))