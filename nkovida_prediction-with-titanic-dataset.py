%matplotlib notebook

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# load data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info(), test.info()
# set "PassengerId" variable as index

train.set_index("PassengerId", inplace=True)

test.set_index("PassengerId", inplace=True)
# generate training target set (y_train)

y_train = train["Survived"]
# delete column "Survived" from train set

train.drop(labels="Survived", axis=1, inplace=True)
# shapes of train and test sets

train.shape, test.shape
# join train and test sets to form a new train_test set

train_test =  train.append(test)
# delete columns that are not used as features for training and prediction

columns_to_drop = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]

train_test.drop(labels=columns_to_drop, axis=1, inplace=True)
# convert objects to numbers by pandas.get_dummies

train_test_dummies = pd.get_dummies(train_test, columns=["Sex"])
# check the dimension

train_test_dummies.shape
# replace nulls with 0.0

train_test_dummies.fillna(value=0.0, inplace=True)
# generate feature sets (X)

X_train = train_test_dummies.values[0:891]

X_test = train_test_dummies.values[891:]
X_train.shape, X_test.shape
# transform data

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scale = scaler.fit_transform(X_train)

X_test_scale = scaler.transform(X_test)
# split training feature and target sets into training and validation subsets

from sklearn.model_selection import train_test_split



X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, y_train, random_state=0)
# import machine learning algorithms

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# train with Gradient Boosting algorithm

# compute the accuracy scores on train and validation sets when training with different learning rates



learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in learning_rates:

    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)

    gb.fit(X_train_sub, y_train_sub)

    print("Learning rate: ", learning_rate)

    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_sub, y_train_sub)))

    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation_sub, y_validation_sub)))

    print()
# Output confusion matrix and classification report of Gradient Boosting algorithm on validation set



gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)

gb.fit(X_train_sub, y_train_sub)

predictions = gb.predict(X_validation_sub)



print("Confusion Matrix:")

print(confusion_matrix(y_validation_sub, predictions))

print()

print("Classification Report")

print(classification_report(y_validation_sub, predictions))
# ROC curve and Area-Under-Curve (AUC)



y_scores_gb = gb.decision_function(X_validation_sub)

fpr_gb, tpr_gb, _ = roc_curve(y_validation_sub, y_scores_gb)

roc_auc_gb = auc(fpr_gb, tpr_gb)



print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
logreg = LogisticRegression()

logreg.fit(X_train_sub, y_train_sub)

predictions = logreg.predict(X_validation_sub)

print("Confusion Matrix:")

print(confusion_matrix(y_validation_sub, predictions))

print()

print("Classification Report")

print(classification_report(y_validation_sub, predictions))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train_sub, y_train_sub)



predictions = random_forest.predict(X_validation_sub)

print("Confusion Matrix:")

print(confusion_matrix(y_validation_sub, predictions))

print()

print("Classification Report")

print(classification_report(y_validation_sub, predictions))