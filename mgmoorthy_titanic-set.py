# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load data

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
train.head()
test.head()
train.info(), test.info()
# set "PassengerId" variable as index

train.set_index("PassengerId", inplace=True)

test.set_index("PassengerId", inplace=True)
train.shape
test.shape
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
train_test.head()
# convert objects to numbers by pandas.get_dummies

train_test_dummies = pd.get_dummies(train_test, columns=["Sex"])
train_test_dummies.head()
# check the dimension

train_test_dummies.shape
train_test_dummies.isnull().sum()
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

X_test_scale = scaler.fit_transform(X_test)
# split training feature and target sets into training and validation subsets

from sklearn.model_selection import train_test_split



X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, y_train, random_state=0)
# import machine learning algorithms

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# train with Gradient Boosting algorithm

# compute the accuracy scores on train and validation sets when training with different learning rates



learning_rates = [0.05, 0.1,0.001, 0.25, 0.5, 0.75, 1]

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
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(X_train_sub, y_train_sub)
xgb.score(X_validation_sub, y_validation_sub)