# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Keep random values consistent

np.random.seed(0)
# Read in the train and test data into a pandas dataframe

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Gather brief overview of training data

train.head()
# Set sample ids as dataframe indices for train, and test sets
train.set_index("Id", inplace=True)
test.set_index("Id", inplace=True)
# Since most features consist of string values which can't be read into a machine 
# learning algorithm, let's convert them all to indicator values

train = pd.get_dummies(train)
test = pd.get_dummies(test)
# Check to see if any features are missing values and view some of those feature's values

for feature in train.columns:
    if train[feature].isnull().any():
        print(train[feature].head())
# Modify training and test data so that any missing values are converted to the corresponding 
# feature's median value

train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)
# Confirm how many features we have

print("Number of features:", train.shape[1])
# To minimize the number of features, let's identify all those with a p-value <= 0.05

from scipy.stats import pearsonr

relevant_features = []
for feature in train.columns:
    if feature != "SalePrice":
        p_value = pearsonr(train[feature], train["SalePrice"])[1]
        if p_value <= 0.05:
            relevant_features.append(feature)
print("Number of relevant features:", len(relevant_features))
# Randomly partition our training set into a training and development set 
# with a 9:1 ratio

train_shuffled = train.sample(frac=1, random_state=0)
mask = np.random.rand(len(train)) < 0.9
train = train_shuffled[mask]
dev = train_shuffled[~mask]
# Partition train, dev, and test sets into predictors and responses, 
# only using features identified as relevant as predictors
X_train = train[relevant_features]
Y_train = train["SalePrice"]
X_dev = dev[relevant_features]
Y_dev = dev["SalePrice"]
X_test = test[relevant_features]
# Train several different regressors on our training data and aggregate them to a list

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rr_model = Ridge(random_state=0).fit(X_train, Y_train)
svr_linear_model = SVR(kernel="linear").fit(X_train, Y_train)
svr_rbf_model = SVR(kernel="rbf").fit(X_train, Y_train)
rfr_model = RandomForestRegressor(random_state=0).fit(X_train, Y_train)
gbr_model = GradientBoostingRegressor(random_state=0).fit(X_train, Y_train)

models = [rr_model, svr_linear_model, svr_rbf_model, rfr_model, gbr_model]
# Identify the "chosen" model out of our collection of models as the one that has the 
# highest R^2 score on the development set

chosen_model = None
highest_score = 0

for model in models:
    score = model.score(X_dev, Y_dev)
    if score > highest_score:
        chosen_model = model
        highest_score = score
# Submit chosen model predictions on test set

predictions = [{"Id": sample, 
                "SalePrice": chosen_model.predict(X_test.loc[sample]
                                                        .values
                                                        .reshape(1, len(X_test.loc[sample])))[0]} 
               for sample in X_test.index]
submission = pd.DataFrame(predictions)
submission.to_csv("submission.csv", index=False)
