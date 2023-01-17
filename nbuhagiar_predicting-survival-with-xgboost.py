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
# Set seed so that results are reproducible

np.random.seed(0)
# Place train and test data into pandas dataframes

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Observe some train data observations

train.head()
# Observe some test data observations

test.head()
# Gather brief overview of train data

train.describe(include="all")
# Gather brief overview of test data

test.describe(include="all")
# Set the PassengerId to be the index for both the train and test sets

train.set_index("PassengerId", inplace=True)
test.set_index("PassengerId", inplace=True)
# Due to the arbitrariness of the values present in the Ticket feature, and 
# the number of missing values in the Cabin feature, both shall be removed 
# from the train and test sets. Where a passenger embarked on intuitively 
# wouldn't affect whether or not they survive, so that feature will also be 
# removed

train.drop("Ticket", axis=1, inplace=True)
train.drop("Cabin", axis=1, inplace=True)
train.drop("Embarked", axis=1, inplace=True)

test.drop("Ticket", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)
test.drop("Embarked", axis=1, inplace=True)
# Due to the number of possible names, the Name feature probably won't yield helpful information 
# with regards to who survives this disaster, and using it as a model feature will probably lead 
# to us overfitting the data. The title in the name may be useful however in giving additional 
# information about a given observation, so we will extract that feature and drop the rest of the 
# information with regards to a person's name

train.drop("Name", axis=1, inplace=True)
test.drop("Name", axis=1, inplace=True)
# We will set the Age and Fare for any null values to be the Age and Fare median 
# respectively in the train and test sets

train["Age"].fillna(train["Age"].median(), inplace=True)
train["Fare"].fillna(train["Fare"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)
test["Fare"].fillna(test["Fare"].median(), inplace=True)
# Let's transform all categorical features to dummy values

train = pd.get_dummies(train)
test = pd.get_dummies(test)
# Check to see if this is a class imblance problem

train["Survived"].value_counts()
# Parition training set into train and dev set in several different ways and train a 
# linear SVM and gradient boosting classifier on each one to see which 
# machine learning model performs the best

from sklearn.model_selection import ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

best_model = None
highest_score = 0

shuffler = ShuffleSplit(test_size=0.33, random_state=0)
for train_index, dev_index in shuffler.split(train):
    train_instance = train.iloc[train_index]
    X_train = train_instance.drop(labels=["Survived"], axis=1)
    Y_train = train_instance["Survived"]
    dev_instance = train.iloc[dev_index]
    X_dev = dev_instance.drop(labels=["Survived"], axis=1)
    Y_dev = dev_instance["Survived"]
    svc_linear = LinearSVC(random_state=0)
    rf = RandomForestClassifier(random_state=0)
    gb = GradientBoostingClassifier(random_state=0)
    xgb = XGBClassifier(random_state=0)
    models = [svc_linear, rf, gb, xgb]
    for model in models:
        model.fit(X_train, Y_train)
        score = model.score(X_dev, Y_dev)
        if score > highest_score:
            best_model = model
            highest_score = score
    
print(best_model, highest_score)
"""# Fine-tune XGB classifier hyperparameters to yield the model
# that best fits the data

best_n_estimators = None
best_lr = None
highest_score = 0

shuffler = ShuffleSplit(test_size=0.33, random_state=0)
for train_index, dev_index in shuffler.split(train):
    train_instance = train.iloc[train_index]
    X_train = train_instance.drop(labels=["Survived"], axis=1)
    Y_train = train_instance["Survived"]
    dev_instance = train.iloc[dev_index]
    X_dev = dev_instance.drop(labels=["Survived"], axis=1)
    Y_dev = dev_instance["Survived"]
    for num in [10, 50, 100, 500, 1000]:
        for lr in [0.001, 0.01, 0.1]:
            model = XGBClassifier(learning_rate=lr, n_estimators=num, random_state=0)
            model.fit(X_train, Y_train, early_stopping_rounds=5, eval_set=[(X_dev, Y_dev)], verbose=False)
            score = model.score(X_dev, Y_dev)
            if score >= highest_score:
                best_n_estimators = num
                best_lr = lr
                highest_score = score
    
print(best_n_estimators, best_lr, highest_score)"""
# Train classifier using fine-tuned parameters

X_train = train.drop(labels=["Survived"], axis=1)
Y_train = train["Survived"]
classifier = XGBClassifier(n_estimators=1000, learning_rate=0.001, random_state=0)
classifier.fit(X_train, Y_train)
# Submit chosen model predictions on test set

predictions = classifier.predict(test)
submission = pd.DataFrame(predictions).reset_index()
submission.columns = ["PassengerId", "Survived"]
submission["PassengerId"] += 892
submission.to_csv("submission.csv", index=False)
