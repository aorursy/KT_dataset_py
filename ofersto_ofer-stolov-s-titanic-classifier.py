# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import cross_validation
def create_submission(alg, train, test, features, filename):

    alg.fit(train[features], train["Survived"])

    predictions = alg.predict(test[features])



    submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

    

    submission.to_csv(filename, index=False)
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
# Sex to number

train_df["Sex"] = train_df["Sex"].map({"female": 0, "male": 1}).astype(int)

test_df["Sex"] = test_df["Sex"].map({"female": 0, "male": 1}).astype(int)
# Empty ages and fares to median

age_median = train_df.append(test_df)["Age"].dropna().median()

train_df["Age"].fillna(age_median, inplace=True)

test_df["Age"].fillna(age_median, inplace=True)



fare_median = train_df.append(test_df)["Fare"].dropna().median()

train_df["Fare"].fillna(age_median, inplace=True)

test_df["Fare"].fillna(age_median, inplace=True)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(

    n_estimators=250,

    random_state=1,

    min_samples_leaf=2,

)

scores = cross_validation.cross_val_score(rf_clf, train_df[features], train_df["Survived"], cv=3)

print(scores.mean())

create_submission(rf_clf, train_df, test_df, features, "random_forest.csv")
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()

scores = cross_validation.cross_val_score(rf_clf, train_df[features], train_df["Survived"], cv=3)

print(scores.mean())

create_submission(gb_clf, train_df, test_df, features, "gradient_boosting.csv")
from sklearn.ensemble import AdaBoostClassifier

ab_clf = AdaBoostClassifier(

    learning_rate=0.1,

    n_estimators=250

)

scores = cross_validation.cross_val_score(ab_clf, train_df[features], train_df["Survived"], cv=3)

print(scores.mean())

create_submission(gb_clf, train_df, test_df, features, "ada_boosting.csv")
from sklearn.neural_network import MLPClassifier

mlp_clf = MLPClassifier(

    max_iter=500

)

scores = cross_validation.cross_val_score(mlp_clf, train_df[features], train_df["Survived"], cv=3)

print(scores.mean())

create_submission(mlp_clf, train_df, test_df, features, "mlp_classifier.csv")
from sklearn.neighbors import KNeighborsClassifier

kn_clf = KNeighborsClassifier()

scores = cross_validation.cross_val_score(kn_clf, train_df[features], train_df["Survived"], cv=3)

print(scores.mean())

create_submission(kn_clf, train_df, test_df, features, "k_neighbors.csv")