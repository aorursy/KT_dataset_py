import numpy as np

import matplotlib.pyplot as ply

import seaborn as sns

import pylab as plot

import pandas_profiling

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.profile_report()
train.describe()
passengerId = test.PassengerId

print(test)

titanic = train.append(test, ignore_index=True)

train_idx = len(train)

test_idx = len(titanic) - len(test)
titanic.head()
titanic.info()
titanic['Title'] = titanic.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}



titanic.Title = titanic.Title.map(normalized_titles)



print(titanic.Title.value_counts())
grouped = titanic.groupby(['Sex','Pclass', 'Title']) 

grouped.Age.median()
titanic.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))



titanic.Cabin = titanic.Cabin.fillna('U')



most_embarked = titanic.Embarked.value_counts().index[0]



titanic.Embarked = titanic.Embarked.fillna(most_embarked)



titanic.Fare = titanic.Fare.fillna(titanic.Fare.median())



titanic.info()
titanic['FamilySize'] = titanic.Parch + titanic.SibSp + 1



titanic.Cabin = titanic.Cabin.map(lambda x: x[0])
titanic.Sex = titanic.Sex.map({"male": 0, "female":1})



pclass_dummies = pd.get_dummies(titanic.Pclass, prefix="Pclass")

title_dummies = pd.get_dummies(titanic.Title, prefix="Title")

cabin_dummies = pd.get_dummies(titanic.Cabin, prefix="Cabin")

embarked_dummies = pd.get_dummies(titanic.Embarked, prefix="Embarked")



titanic_dummies = pd.concat([titanic, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)





titanic_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

titanic_dummies.head()
train = titanic_dummies[ :train_idx]

test = titanic_dummies[test_idx: ]



train.Survived = train.Survived.astype(int)



X = train.drop('Survived', axis=1).values 

y = train.Survived.values



X_test = test.drop('Survived', axis=1).values

forrest_params = dict(     

    max_depth = [n for n in range(9,14)],     

    min_samples_split = [n for n in range(4, 12)], 

    min_samples_leaf = [n for n in range(4, 14)],     

    n_estimators = [n for n in range(10, 100, 5)],

)
forest = RandomForestClassifier()
forest_cv = GridSearchCV(estimator=forest, param_grid=forrest_params, cv=5) 

forest_cv.fit(X, y)
print("Best score: {}".format(forest_cv.best_score_))

print("Optimal params: {}".format(forest_cv.best_estimator_))
forrest_pred = forest_cv.predict(X_test)

kaggle = pd.DataFrame({'PassengerId': passengerId, 'Survived': forrest_pred})

kaggle.to_csv('../working/titanic.csv', index=False)
from xgboost import XGBClassifier
xg = XGBClassifier(learning_rate=0.02, n_estimators=750,

                   max_depth= 3, min_child_weight= 1, 

                   colsample_bytree= 0.6, gamma= 0.0, 

                   reg_alpha= 0.001, subsample= 0.8

                  )

xg.fit(X, y)

xg_predictions = xg.predict(X_test)

xg_data = test

xg_data.to_csv('../working/XGBoost_SS_OH_FE_GSCV.csv')
from sklearn.ensemble import RandomForestRegressor 

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X, y)

y_pred = random_forest.predict(X_test)

print(random_forest.score(X, y))

y_pred = pd.DataFrame(data=y_pred)
y_pred.to_csv("../working/Pred.csv")