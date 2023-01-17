# Titanic Machine Learning

# Data preparation, grid search for best parameters and Random Forest classifier application
import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV 

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# data upload

titanc_gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

titanic_test = pd.read_csv("../input/titanic/test.csv")

titanic_train = pd.read_csv("../input/titanic/train.csv")



# index for submission

Id = titanic_test.PassengerId



# test response

y = titanic_train.Survived
# view of data in train dataset

titanic_train.head()
# train daset description: checking ranges of predictors and missing data

titanic_train.describe()
# test daset description: checking ranges of predictors and missing data

titanic_test.describe()
# fill missing data NA in Age an Fare with mean value

titanic_train = titanic_train.fillna(titanic_train.mean())

titanic_test = titanic_test.fillna(titanic_test.mean())
# Data reparation

# categorical variable becomes numerical: using qcut to assing numerical labels

# new features
# Family feature

titanic_train["Family"] = titanic_train.SibSp + titanic_train.Parch + 1

titanic_test["Family"] = titanic_test.SibSp + titanic_test.Parch + 1
# create categories for Age

age_labels = [0, 1, 2, 3, 4]

titanic_train["Age_cat"] = pd.qcut(titanic_train["Age"], q = 5, labels=age_labels)

titanic_test["Age_cat"] = pd.qcut(titanic_train["Age"], q = 5, labels=age_labels)
# create categories for Sex

titanic_train['Sex'] = titanic_train['Sex'].replace(['female'], 1)

titanic_train['Sex'] = titanic_train['Sex'].replace(['male'], 2)

titanic_test['Sex'] =  titanic_test['Sex'].replace(['female'], 1)

titanic_test['Sex'] =  titanic_test['Sex'].replace(['male'], 2)
# create Title feature

titanic_train["Title"] = titanic_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

titanic_test["Title"] = titanic_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# create categories for Title

# train set

titanic_train['Title'] = titanic_train['Title'].replace(['Dr', 'Don', 'Col', 'Capt', 'Jonkheer', 'Major', 'Rev', 'Countess', 'Sir', 'Lady'], 0)

titanic_train['Title'] = titanic_train['Title'].replace(['Mrs', 'Mme', 'Dona', 'Ms'], 1)

titanic_train['Title'] = titanic_train['Title'].replace(['Mlle', 'Miss'], 2)

titanic_train['Title'] = titanic_train['Title'].replace(['Mr', 'Master'], 3)
# test set

titanic_test['Title'] = titanic_test['Title'].replace(['Dr', 'Don', 'Col', 'Capt', 'Jonkheer', 'Major', 'Rev', 'Countess', 'Sir', 'Lady'], 0)

titanic_test['Title'] = titanic_test['Title'].replace(['Mrs', 'Mme', 'Dona', 'Ms'], 1)

titanic_test['Title'] = titanic_test['Title'].replace(['Mlle', 'Miss'], 2)

titanic_test['Title'] = titanic_test['Title'].replace(['Mr', 'Master'], 3)
# drop redundant features from train set

titanic_train = titanic_train.drop(["PassengerId", "Ticket", "Cabin", "Fare", "Age", "Name", "Embarked", "SibSp", "Parch", "Survived"], axis=1)
# drop redundant features from test set

titanic_test = titanic_test.drop(["PassengerId", "Ticket", "Cabin", "Fare", "Age", "Name", "Embarked", "SibSp", "Parch"], axis=1)
# check train set is ready for classification

titanic_train.head()
# test set is ready for classification

titanic_test.head()
# split data for train and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(titanic_train, y, test_size=0.2, random_state=11)
# model definition

model = RandomForestClassifier()
# parameters grid

param = {'n_estimators': [4, 6, 8, 10], 

         'max_depth': [3, 5, 7], 

         'min_samples_split': [2, 3, 4],

         'min_samples_leaf': [3, 5, 7],

         'max_leaf_nodes': [3, 5, 7],

         'random_state': [102]

         }
# set grid search

scorer = make_scorer(accuracy_score)

grid = GridSearchCV(model, param, scoring=scorer)

grid_fit = grid.fit(X_train, y_train)
# selecting best model

best_model = grid_fit.best_estimator_
# fitting best model with full train set

best_model.fit(X_train, y_train)
# feature importance

best_model.feature_importances_
# prediction on validation set and accuracy score on predictions

preds = best_model.predict(X_valid)

print(accuracy_score(y_valid, preds))



# removing Age_cat from the model does not improve the accuracy score
# making predictions on test set

preds_2 = best_model.predict(titanic_train)
# fitting best model with full train set

best_model.fit(titanic_train, y)

print(accuracy_score(y, preds_2))
# making predictions on test set

preds_submission = best_model.predict(titanic_test)
# preparing csv file for submission

output = pd.DataFrame({'PassengerId': Id, 'Survived': preds_submission})

output.to_csv('Titanic_submission.csv', index=False)
# check output shape

output.shape