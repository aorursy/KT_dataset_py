# Importing all necessary Python Libraries.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Uploading the training and testing datasets from Kaggle.

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.info()
# Checking for missing values.

train_data.isnull().sum()
# Removing the Cabin feature due to its extensive missing values.

train_data.drop('Cabin', axis=1, inplace=True)
# Replacing the missing embarked values with the value that appears the most frequently.

train_data["Embarked"].fillna(train_data['Embarked'].value_counts().idxmax(), inplace=True)
# Replacing the missing age values with the median age value.

train_data["Age"].fillna(train_data["Age"].median(skipna=True), inplace=True)
train_data.isnull().sum()
train_data.head()
# Encoding the sex values.

train_data['Sex'].replace("female", 0,inplace=True)

train_data['Sex'].replace("male", 1,inplace=True)
# Encoding the embarked values.

train_data['Embarked'].replace("S", 0,inplace=True)

train_data['Embarked'].replace("C", 1,inplace=True)

train_data['Embarked'].replace("Q", 2,inplace=True)
train_data.dtypes
test_data.isnull().sum()
test_data["Age"].fillna(test_data["Age"].median(skipna=True), inplace=True)

test_data["Fare"].fillna(test_data["Fare"].median(skipna=True), inplace=True)

test_data.drop('Cabin', axis=1, inplace=True)
test_data['Sex'].replace("female", 0,inplace=True)

test_data['Sex'].replace("male", 1,inplace=True)
test_data['Embarked'].replace("S", 0,inplace=True)

test_data['Embarked'].replace("C", 1,inplace=True)

test_data['Embarked'].replace("Q", 2,inplace=True)
test_data.dtypes
train_data.shape
test_data.shape
train_data.head()
outcome_data = train_data["Survived"]

train_data.drop(["Survived", "Ticket", "Name", "PassengerId"], axis=1, inplace=True)

test_data.drop(["Name","PassengerId","Ticket"], axis=1, inplace=True)
from sklearn.model_selection import train_test_split



# Selecting the features and the outcome.

X = train_data.values

y = outcome_data.values



# Splitting the data into training and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
# Initializing a RandomForestClassifier.

rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=5, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=350,

                       n_jobs=None, oob_score=False, random_state=1

                            , verbose=0,

                       warm_start=False)



rf.fit(X_train, y_train)



# Predicting from the test set.

y_pred = rf.predict(X_test)



# Predicting from the train set.

y_pred_train = rf.predict(X_train)



# Printing the accuracy with accuracy_score function.

print("Accuracy Train: ", accuracy_score(y_train, y_pred_train))



# Printing the accuracy with accuracy_score function.

print("Accuracy Test: ", accuracy_score(y_test, y_pred))



# Printing the confusion matrix.

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))
last_clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=5, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=350,

                       n_jobs=None, oob_score=True, random_state=1, verbose=0,

                       warm_start=False)



last_clf.fit(train_data, outcome_data)

print("%.4f" % last_clf.oob_score_)
ids = pd.read_csv("/kaggle/input/titanic/test.csv")[["PassengerId"]].values



# Making predictions.

predictions = last_clf.predict(test_data.values)



# Printing the predictions.

print(predictions)



# Creating a dictionary with passenger ids and predictions.

df = {'PassengerId': ids.ravel(), 'Survived':predictions}



# Creating a DataFrame named submission.

submission = pd.DataFrame(df)



# Displaying the first five rows of submission.

display(submission.head())



# Saving the file.

submission.to_csv("submission.csv", index=False)