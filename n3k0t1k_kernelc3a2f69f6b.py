import utils

import numpy as np

import pandas as pd

from sklearn import tree, model_selection



import matplotlib

import matplotlib.pyplot as plt



%matplotlib inline
# Reading Train, test Data

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

train.head()
# Comparing survivors vs deceased

train.Survived.value_counts(normalize=True).plot(kind='bar')
# Kind of ticket First Class, Second Class, Third Class

train.Pclass.value_counts(normalize=True).plot(kind="bar")
# Where the people boarded { C = Cherbourg, Q = Queenstown, S = Southampton }

train.Embarked.value_counts(normalize=True).plot(kind='bar')
# Women that survided vs deceased

train.Survived[train.Sex == "male"].value_counts(normalize=True).plot(kind='bar')
# Men that survided vs deceased

train.Survived[train.Sex == "female"].value_counts(normalize=True).plot(kind='bar')
# Survived - Men vs Woman

train[train.Survived == 1].Sex.value_counts(normalize=True).plot(kind='bar')
# Men with 3rd class ticket

train.Survived[(train.Sex == "male") & (train.Pclass == 3)].value_counts(normalize=True).plot(kind='bar')
# Men with 1st class ticket

train.Survived[(train.Sex == "male") & (train.Pclass == 1)].value_counts(normalize=True).plot(kind='bar')
# Women with 3rd class ticket

train.Survived[(train.Sex == "female") & (train.Pclass == 3)].value_counts(normalize=True).plot(kind='bar')
# Women with 1st class ticket

train.Survived[(train.Sex == "female") & (train.Pclass == 1)].value_counts(normalize=True).plot(kind='bar')
train["hypotesis"] = 0

# My hypotesis about woman have more chance to survive

train.loc[train.Sex == "female", "hypotesis"] = 1



train["result"] = 0

# counting survivors

train.loc[train.Survived == train["hypotesis"], "result"] = 1



train["result"].value_counts(normalize=True)
# Cleaning data

utils.clean_data(train)

utils.clean_data(test)
# Getting relevant values

target = train["Survived"].values

features = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
from sklearn import linear_model, preprocessing, model_selection



logistic_algotithm = linear_model.LogisticRegression()

logistic_algotithm.fit(features, target)



print(logistic_algotithm.score(features, target))
poly = preprocessing.PolynomialFeatures(degree=2)

poly_features = poly.fit_transform(features)
logistic_algotithm.fit(poly_features, target)

print(logistic_algotithm.score(poly_features, target))
# Cross valdiation

scores = model_selection.cross_val_score(logistic_algotithm, poly_features, target, scoring='accuracy', cv=10)

print(scores.mean())
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

test_features_ = poly.fit_transform(test_features)

utils.write_prediction(logistic_algotithm.predict(test_features_), "gender_submission.csv")