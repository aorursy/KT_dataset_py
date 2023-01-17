# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info()

print("-"*40)

test.info()
train.describe(include="all")
train.head()
train_age_missing = train[pd.isnull(train.Age)]

for name in train_age_missing.Name:

    if "miss" in name.lower():

        train_age_missing.Age = 20.0

    else:

        train_age_missing.Age = 30.0

train_age_non_missing = train[pd.notnull(train.Age)]

train_model = pd.concat([train_age_missing,train_age_non_missing])



test_age_missing = test[pd.isnull(test.Age)]

for name in test_age_missing.Name:

    if "miss" in name.lower():

        test_age_missing.Age = 20.0

    else:

        test_age_missing.Age = 30.0

test_age_non_missing = test[pd.notnull(test.Age)]

test = pd.concat([test_age_missing,test_age_non_missing])

def Sex_to_number(sex) :

    if sex=="male":

        return 1

    else:

        return 0

train_model["Sex_n"] = train_model.apply(lambda x : Sex_to_number(x.Sex),axis=1)

test["Sex_n"] = test.apply(lambda x : Sex_to_number(x.Sex),axis=1)
train_model = train_model[["Survived","Pclass","Sex_n","Age"]]

test = test[["PassengerId","Pclass","Sex_n","Age"]]
X_train_model = train_model.drop("Survived",axis = 1)

Y_train_model = train_model.Survived

X_test = test[["Pclass","Sex_n","Age"]]
# logistic regression

from sklearn.linear_model import LogisticRegression 

logreg = LogisticRegression()

logreg.fit(X_train_model, Y_train_model)



acc_log = round(logreg.score(X_train_model, Y_train_model) * 100, 2)

acc_log
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train_model, Y_train_model)



acc_decision_tree = round(decision_tree.score(X_train_model, Y_train_model) * 100, 2)

acc_decision_tree
# Random Forest

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train_model, Y_train_model)



acc_random_forest = round(random_forest.score(X_train_model, Y_train_model) * 100, 2)

acc_random_forest
Y_pred = random_forest.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    }).sort_values("PassengerId")
submission.to_csv("submission.csv")