import xgboost as xgb

import pandas as pd

from sklearn.preprocessing import LabelEncoder

import numpy as np
titanic = pd.read_csv('../input/train.csv', header=0)

titanic_test = pd.read_csv('../input/test.csv', header=0)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Replace all the occurences of male with the number 0.

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

# Replace all the occurences of female with the number 1.

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1



titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2



# Repeat with test dataset

titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")



titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
train_X = titanic[predictors].as_matrix()

test_X = titanic_test[predictors].as_matrix()

train_y = titanic["Survived"]

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

predictions = gbm.predict(train_X)

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)



#

predictions = gbm.predict(test_X)

submission = pd.DataFrame({ 'PassengerId': titanic_test['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submission.csv", index=False)