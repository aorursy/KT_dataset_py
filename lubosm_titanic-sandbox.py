# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv



from sklearn.ensemble import RandomForestClassifier



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv" )

test = pd.read_csv("../input/test.csv")



train = train.drop("PassengerId", axis=1)



test["Fare"] = test["Fare"].fillna(50)

train["Embarked"] = train["Embarked"].fillna("C")



train.info()

test.info()



train_embark = pd.get_dummies(train["Embarked"])

test_embark = pd.get_dummies(test["Embarked"])



train_pclass = pd.get_dummies(train["Pclass"])

train_pclass.columns = ['C1', 'C2', 'C3']

test_pclass = pd.get_dummies(test["Pclass"])

test_pclass.columns = ['C1', 'C2', 'C3']



train_embark.info()



train = train.join(train_embark)

test = test.join(test_embark)



train = train.join(train_pclass)

test = test.join(test_pclass)



X_train = train[["C1", "C2"]]#train[["Pclass", "Fare", "C", "SibSp", "Parch"]]

Y_train = train["Survived"]

X_test  = test[["C1", "C2"]]#test[["Pclass", "Fare", "C", "SibSp", "Parch"]]



#X_train = X_train[:, None]

#X_test = X_test[:, None]



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)



# output file

output = pd.DataFrame({"PassengerId": test["PassengerId"],

                      "Survived": Y_pred})

output.to_csv("output.csv", index=False)