# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_df.head()
# Set multiple plots distribution
figs, axs = plt.pyplot.subplots(nrows = 3, ncols = 3, figsize=(30,15))

sns.barplot(x="Pclass", y="Survived", data=train_df, ax = axs[0][0])
sns.barplot(x="Sex", y="Survived", data=train_df, ax = axs[0][1])
sns.barplot(x="Parch", y="Survived", data=train_df, ax = axs[0][2])
sns.barplot(x="SibSp", y="Survived", data=train_df, ax = axs[1][0])
sns.boxplot(x="Survived", y="Fare", data=train_df[(train_df["Sex"] == "female") & (train_df["Fare"] < 100)], ax = axs[1][1])
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=train_df[(train_df["Fare"] < 60) & (train_df["Age"] < 60)], ax = axs[1][2])
sns.barplot(x="Embarked", y="Survived", data=train_df, ax=axs[2][1])

def cab_to_deck(cab):
    if type(cab) is float:
        return "N"
    else:
        return cab[0]
    
train_df["Deck"] = train_df["Cabin"].apply(cab_to_deck)
test["Deck"] = test["Cabin"].apply(cab_to_deck)
cabin_is_nan = train_df["Cabin"].isna().sum() / len(train_df["Cabin"])
print("Percentage of NaN: ", cabin_is_nan*100)

figs, axs = plt.pyplot.subplots(ncols = 2, figsize = (30,5))

sns.barplot(x="Deck", y="Survived", data=train_df, ax = axs[0])
sns.countplot(x="Deck", data=train_df, ax = axs[1])
# Criar a Feature
train_df["AgeIsNaN"] = train_df["Age"].isna()
test["AgeIsNaN"] = test["Age"].isna()
sns.barplot(x="AgeIsNaN", y="Survived", data=train_df)
# Substituir valores NaN em Age
train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
test["Age"].fillna(test["Age"].mean(), inplace=True)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# Apagar antigas features convertidas, ticket, Cabin e Name
train_df = train_df.drop(columns = ['Ticket', 'Cabin', 'Name'])
test = test.drop(columns = ['Ticket', 'Cabin', 'Name'])
# Estabelecer um máximo para Fare, 150, para ter melhores resultados ao usar MinMaxScaler
train_df.loc[train_df["Fare"] > 150, "Fare"] = train_df.loc[train_df["Fare"] > 150, "Fare"].apply(lambda x: 100)
test.loc[test["Fare"] > 150, "Fare"] = test.loc[train_df["Fare"] > 150, "Fare"].apply(lambda x: 100)

test["Fare"].fillna(test["Fare"].mean(), inplace = True)

sns.distplot(train_df["Fare"])
scaler = MinMaxScaler()
train_df[["Age", "Fare"]] = scaler.fit_transform(train_df[["Age", "Fare"]])
test[["Age", "Fare"]] = scaler.fit_transform(test[["Age", "Fare"]])
train_df['Embarked'].fillna('C', inplace=True)
test['Embarked'].fillna('C', inplace=True)
encoder = LabelEncoder()

train_df[['Sex', 'Embarked', 'Deck']] = train_df[['Sex', 'Embarked', 'Deck']].apply(lambda feat: encoder.fit_transform(feat))
test[['Sex', 'Embarked', 'Deck']] = test[['Sex', 'Embarked', 'Deck']].apply(lambda feat: encoder.fit_transform(feat))
# Separar as labels dos dados e retirar os ids
ids = test["PassengerId"]
y = train_df["Survived"]
X = train_df.drop(columns = ["Survived", "PassengerId"])
test = test.drop(columns = ["PassengerId"])
# Separar os dados em casos de teste e de treino
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
print(decision_tree.score(X_train, y_train))
print(cross_val_score(decision_tree, X, y).mean())
parameters = {'criterion' : ['gini', 'entropy'],
              'splitter' : ['best', 'random'],
              'max_depth': range(1, 15),
              'min_samples_leaf' : range(1, 100, 10),
             }

cv = GridSearchCV(decision_tree, parameters)

cv.fit(X, y)

print(cv.best_score_)
cv.best_params_
tree_labels = cv.best_estimator_.predict(test)
submission = pd.DataFrame(data= {'PassengerId' : ids, "Survived" : tree_labels})
submission.to_csv('DecisionTree.csv', index=False)
random_forest = RandomForestClassifier(n_estimators = 1000, oob_score = True)
random_forest.fit(X_train, y_train)
print(random_forest.score(X_train, y_train))
print(cross_val_score(random_forest, X, y).mean())
parameters = {
    'n_estimators' : [200],
    'max_depth' : [8],
    'max_features' : range(1, 10),
} 

cv = GridSearchCV(random_forest, parameters)

cv.fit(X, y)

print(cv.best_score_)
best_forest = RandomForestClassifier(n_estimators = 1400, max_depth = 8, max_features = 7)
best_forest.fit(X_train, y_train)
forest_labels = cv.best_estimator_.predict(test)
submission = pd.DataFrame(data= {'PassengerId' : ids, "Survived" : forest_labels})
submission.to_csv('RandomForest.csv', index=False)
grad_boost = GradientBoostingClassifier()
grad_boost.fit(X_train, y_train)
print(grad_boost.score(X_train, y_train))
print(grad_boost.score(X_test, y_test))
gdboost_labels = cv.best_estimator_.predict(test)
submission = pd.DataFrame(data= {'PassengerId' : ids, "Survived" : gdboost_labels})
submission.to_csv('GradientBoosting.csv', index=False)