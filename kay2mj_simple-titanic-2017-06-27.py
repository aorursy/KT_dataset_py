import pandas as pd

import numpy as np
# import test data

df1 = pd.read_csv("../input/train.csv")
df1 = df1.replace(["male", "female"], [0,1])

df1 = df1.replace(["S", "C", "Q"], [0,1,2])

df1= df1.fillna(0)
y = df1[["Survived"]]

X = df1[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
# import test data

df2 = pd.read_csv("../input/test.csv")
df2 = df2.replace(["male", "female"], [0,1])

df2 = df2.replace(["S", "C", "Q"], [0,1,2])

df2= df2.fillna(0)
X_test = df2[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
from sklearn.cross_validation import train_test_split
# Data Split

X_train, X_test1, y_train, y_test1 = train_test_split(X, y, train_size=0.3)
## model

# Random Forest

import sklearn.ensemble as ske
reg = ske.RandomForestClassifier(n_estimators=14)
reg.fit(X, y)
reg.score(X, y)
reg.score(X_test1, y_test1)
#Result Print

y_pred = reg.predict(X_test)
# DataFrameに変換

y_pred = pd.DataFrame(y_pred)
result = pd.concat([X_test[["PassengerId"]], y_pred], axis = 1)
result.to_csv("result.csv", index=False)