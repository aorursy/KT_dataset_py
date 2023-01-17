# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import seaborn as sns
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.head(3)
train_data.info()
def extract_ticket(ticket):
    ticket = ticket.replace('.', '').replace('/', '').split()
    if len(ticket) > 1:
        return ticket[0]
    else:
        return "X"

def map_family_size(size):
    if size < 2:
        return "Single"
    elif 2 <= size < 4:
        return "Small"
    else:
        return "Large"

X_train = train_data[["Name", "Sex", "Age", "Pclass", "Fare", "Embarked", "SibSp", "Parch", "Cabin", "Ticket"]]
X_test = test_data[["Name", "Sex", "Age", "Pclass", "Fare", "Embarked", "SibSp", "Parch", "Cabin", "Ticket"]]
y_train = train_data["Survived"]

X_train["Honorific"] = X_train["Name"].map(lambda name: name.split('.')[0].split(', ')[-1])
X_test["Honorific"] = X_test["Name"].map(lambda name: name.split('.')[0].split(', ')[-1])
X_train.drop("Name", axis=1, inplace=True)
X_test.drop("Name", axis=1, inplace=True)

X_train["RelativesCount"] = X_train["SibSp"].combine(X_train["Parch"], lambda x1, x2: x1+x2)
X_test["RelativesCount"] = X_test["SibSp"].combine(X_test["Parch"], lambda x1, x2: x1+x2)
X_train.drop(["SibSp", "Parch"], axis=1, inplace=True)
X_test.drop(["SibSp", "Parch"], axis=1, inplace=True)

X_train["FamilySize"] = X_train["RelativesCount"].add(1).map(map_family_size)
X_test["FamilySize"] = X_test["RelativesCount"].add(1).map(map_family_size)

X_train["Deck"] = X_train["Cabin"].fillna('U').map(lambda s: s[0])
X_test["Deck"] = X_test["Cabin"].fillna('U').map(lambda s: s[0])
X_train.drop("Cabin", axis=1, inplace=True)
X_test.drop("Cabin", axis=1, inplace=True)

X_train["Ticket"] = X_train["Ticket"].map(extract_ticket)
X_train["Age"].fillna(X_train["Age"].median(), inplace=True)
X_test["Age"].fillna(X_test["Age"].median(), inplace=True)

X_train["Embarked"].fillna(X_train["Embarked"].mode().values[0], inplace=True)
X_test["Embarked"].fillna(X_train["Embarked"].mode().values[0], inplace=True)

X_train["Fare"].fillna(X_train["Fare"].mean(), inplace=True)
X_test["Fare"].fillna(X_train["Fare"].mean(), inplace=True)

X_train["Pclass"] = X_train["Pclass"].astype("str")
X_test["Pclass"] = X_test["Pclass"].astype("str")

X_train = pd.get_dummies(X_train, columns=["Honorific", "Sex", "Pclass", "Embarked", "Deck", "Ticket", "FamilySize"])
X_test = pd.get_dummies(X_test, columns=["Honorific", "Sex", "Pclass", "Embarked", "Deck", "Ticket", "FamilySize"])

extra_columns = set(X_train.columns) - set(X_test.columns)
for col in extra_columns:
    X_test[col] = 0
X_test = X_test[X_train.columns]
X_train.head()
params = {
    "n_estimators": [10, 20, 30, 50, 100, 150, 200],
    "criterion": ["gini", "entropy"],
    "max_features": ["auto", "sqrt", "log2", None],
    "max_depth": list(range(2, 11)) + [None]
}
gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, cv=5, scoring="accuracy")
gs.fit(X_train, y_train)
gs.best_params_
model = RandomForestClassifier(**gs.best_params_)
model.fit(X_train, y_train)
print("Train score:", model.score(X_train, y_train))
print("CV score:", cross_val_score(model, X_train, y_train, scoring="accuracy", cv=5).mean())
y_pred = model.predict(X_test)
result = test_data[["PassengerId"]].assign(Survived=y_pred)
result.head()
result.to_csv("predictions.csv", index=False)
