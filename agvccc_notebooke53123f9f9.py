import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



mark="Survived"

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
def adjust_data(titanic):

   

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    

    

    



    return titanic
train_data = adjust_data(train)

test_data = adjust_data(test)
clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
scores = cross_validation.cross_val_score(

    clf,

    train_data[predictors],

    train_data["Survived"],

    cv=4

)

print(scores.mean())
def create_submission(clf, train, test, predictors, filename):



    clf.fit(train[predictors], train["Survived"])

    predictions = clf.predict(test[predictors])



    submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

    

    submission.to_csv(filename, index=False)
create_submission(clf, train, test, predictors, 'titanic.csv')
print(check_output(["ls", "../input"]).decode("utf8"))