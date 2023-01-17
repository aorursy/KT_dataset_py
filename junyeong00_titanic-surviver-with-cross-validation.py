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
train = pd.read_csv("../input/titanic/train.csv", index_col = "PassengerId")

print(train.shape)

train.head
test = pd.read_csv("../input/titanic/test.csv", index_col = "PassengerId")

print(test.shape)

test.head()
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
# 1. by using plot

sns.countplot(data = train, x = "Sex", hue = "Survived")
# 2. by using pivot

pd.pivot_table(train, index = "Sex", values = "Survived")
sns.countplot(data = train, x = "Pclass", hue = "Survived")
pd.pivot_table(train, index = "Pclass", values = "Survived")
sns.countplot(data = train, x = "Embarked", hue = "Survived")
pd.pivot_table(train, index = "Embarked", values = "Survived")
sns.lmplot(data = train, x = "Age", y = "Fare", hue = "Survived", fit_reg = False)
# delete three outlier data

low_fare = train[train["Fare"] < 500]

train.shape, low_fare.shape
sns.lmplot(data = low_fare, x = "Age", y = "Fare", hue = "Survived", fit_reg = False)
low_fare = train[train["Fare"] < 100]

sns.lmplot(data = low_fare, x = "Age", y = "Fare", hue = "Survived", fit_reg = False)
train.loc[train["Sex"] == "male", "Sex_encode"] = 0

train.loc[train["Sex"] == "female", "Sex_encode"] = 1

train.head()
test.loc[test["Sex"] == "male", "Sex_encode"] = 0

test.loc[test["Sex"] == "female", "Sex_encode"] = 1

test.head()
train[train["Fare"].isnull()]
test[test["Fare"].isnull()]
train["Fare_fillin"] = train["Fare"]

train[["Fare", "Fare_fillin"]].head()
test["Fare_fillin"] = test["Fare"]

test[["Fare", "Fare_fillin"]].head()
test.loc[test["Fare"].isnull(), "Fare_fillin"] = 0



test.loc[test["Fare"].isnull(), ["Fare", "Fare_fillin"]]
train["Embarked_C"] = train["Embarked"] == "C"

train["Embarked_S"] = train["Embarked"] == "S"

train["Embarked_Q"] = train["Embarked"] == "Q"

train[["Embarked_C", "Embarked_S", "Embarked_Q"]].head()
test["Embarked_C"] = test["Embarked"] == "C"

test["Embarked_S"] = test["Embarked"] == "S"

test["Embarked_Q"] = test["Embarked"] == "Q"

test[["Embarked_C", "Embarked_S", "Embarked_Q"]].head()
feature_names = ["Pclass", "Sex_encode", "Fare_fillin", "Embarked_C", "Embarked_S", "Embarked_Q"]

feature_names
label_name = "Survived"
X_train = train[feature_names]

print(X_train.shape)

X_train.head()
X_test = test[feature_names]

print(X_test.shape)

X_test.head()
y_train = train[label_name]

print(y_train.shape)

y_train.head()
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(max_depth = 5)

model
model.fit(X_train, y_train)
import graphviz

from sklearn.tree import export_graphviz



dot_tree = export_graphviz(model,

                           feature_names = feature_names,

                           class_names = ["Perish", "Survived"],

                           out_file = None)





graphviz.Source(dot_tree)
predictions = model.predict(X_test)

print(predictions.shape)

predictions[0:10]
submission = pd.read_csv("../input/titanic/gender_submission.csv", index_col = "PassengerId")

print(submission.shape)

submission.head()
submission["Survived"] = predictions

submission.head()
# submission.to_csv("./decision_tree.csv")

from sklearn.utils.testing import all_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings('ignore')

warnings.warn("FutureWarning")
allAlgorithms = all_estimators(type_filter = "classifier")

allAlgorithms

kfold_cv = KFold(n_splits = 5, shuffle = True)

result = []

for (name, algorithm) in allAlgorithms:

    if(name == 'CheckingClassifier' or name == 'ClassifierChain' or 

       name == 'MultiOutputClassifier' or name == 'OneVsOneClassifier' or 

       name =='OneVsRestClassifier' or name == 'OutputCodeClassifier' or

       name =='VotingClassifier' or name == 'RadiusNeighborsClassifier'): continue

        

    model = algorithm()

    if hasattr(model, "score"):

        scores = cross_val_score(model, X_train, y_train, cv = kfold_cv)

        result.append({"name": name, "mean": np.mean(scores)})



result
import operator

sorted(result, key = operator.itemgetter("mean", "name"))[-5:]
from sklearn.ensemble import GradientBoostingClassifier

bestModel = GradientBoostingClassifier(max_depth = 5)

bestModel.fit(X_train, y_train)
bestPrediction = bestModel.predict(X_test)

print(bestPrediction.shape)



bestSubmission = pd.read_csv("../input/titanic/gender_submission.csv", index_col = "PassengerId")

bestSubmission["Survived"] = bestPrediction

bestSubmission.to_csv("./GradientBoostingClassifier.csv")