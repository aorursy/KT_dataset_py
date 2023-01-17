import numpy as np

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()

test.head()
print(train.shape)

print(test.shape)
def lacking_table(df): 

    null_val = df.isnull().sum()

    percent = 100 * df.isnull().sum()/len(df)

    lacking_table = pd.concat([null_val, percent], axis=1)

    lacking_table_ren_columns = lacking_table.rename(

    columns = {0 : '欠損数', 1 : '%'})

    return lacking_table_ren_columns
lacking_table(train)
lacking_table(test)
train["Age"] = train["Age"].fillna(train["Age"].median())

train["Embarked"] = train["Embarked"].fillna("S")



lacking_table(train)
train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S" ] = 0

train["Embarked"][train["Embarked"] == "C" ] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



train.head()
test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2

test.Fare[152] = test.Fare.median()



test.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
# 説明変数と目的変数の設定

target = train["Survived"].values

features = train[["Pclass", "Sex", "Age",]].values



X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, random_state=30)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
logreg = LogisticRegression().fit(X_train, y_train)



print(logreg.score(X_train, y_train))

print(logreg.score(X_test, y_test))
logreg01 = LogisticRegression(C=0.1).fit(X_train, y_train)

print(logreg01.score(X_train, y_train))

print(logreg01.score(X_test, y_test))
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)

print(logreg001.score(X_train, y_train))

print(logreg001.score(X_test, y_test))
from sklearn.svm import LinearSVC
#LSVC = LinearSVC.fit(X_train, y_train)

#print(LSVC.score(X_train, y_train))

#print(LSVC.score(X_test, y_test))
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=30).fit(X_train, y_train)

print(forest.score(X_train, y_train))

print(forest.score(X_test, y_test))
forest1000 = RandomForestClassifier(n_estimators=1000, random_state=30).fit(X_train, y_train)

print(forest1000.score(X_train, y_train))

print(forest1000.score(X_test, y_test))
forest1 = RandomForestClassifier(n_estimators=100, max_features=3, random_state=30).fit(X_train, y_train)

print(forest1.score(X_train, y_train))

print(forest1.score(X_test, y_test))
forest2 = RandomForestClassifier(n_estimators=100, max_features=1, random_state=30).fit(X_train, y_train)

print(forest2.score(X_train, y_train))

print(forest2.score(X_test, y_test))
forest = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=30).fit(X_train, y_train)

print(forest.score(X_train, y_train))

print(forest.score(X_test, y_test))
from sklearn.ensemble import GradientBoostingClassifier
GBS = GradientBoostingClassifier(random_state=30).fit(X_train, y_train)

print(GBS.score(X_train, y_train))

print(GBS.score(X_test, y_test))

# defaultだと深さ３の決定木が100個で、学習率は0.1
GBS1 = GradientBoostingClassifier(random_state=30, max_depth=1).fit(X_train, y_train)

print(GBS1.score(X_train, y_train))

print(GBS1.score(X_test, y_test))
# n_estimators=1000

GBS2 = GradientBoostingClassifier(random_state=30, n_estimators=1000).fit(X_train, y_train)

print(GBS2.score(X_train, y_train))

print(GBS2.score(X_test, y_test))
# n_estimators=10000

GBS3 = GradientBoostingClassifier(random_state=30, n_estimators=10000).fit(X_train, y_train)

print(GBS3.score(X_train, y_train))

print(GBS3.score(X_test, y_test))
# learning_rate=0.01

GBS4 = GradientBoostingClassifier(random_state=30, learning_rate=0.01).fit(X_train, y_train)

print(GBS4.score(X_train, y_train))

print(GBS4.score(X_test, y_test))
# learning_rate=0.01, n_estimators=10000

GBS5 = GradientBoostingClassifier(random_state=30, learning_rate=0.01, n_estimators=10000).fit(X_train, y_train)

print(GBS5.score(X_train, y_train))

print(GBS5.score(X_test, y_test))
from sklearn.neural_network import MLPClassifier
# defaultで100隠れユニット。

mlp = MLPClassifier(solver='lbfgs', random_state=30).fit(X_train, y_train)

print(mlp.score(X_train, y_train))

print(mlp.score(X_test, y_test))
mlp = MLPClassifier(solver='lbfgs', random_state=30, hidden_layer_sizes=[10,10]).fit(X_train, y_train)

print(mlp.score(X_train, y_train))

print(mlp.score(X_test, y_test))
mlp = MLPClassifier(solver='lbfgs', activation='tanh', random_state=30).fit(X_train, y_train)

print(mlp.score(X_train, y_train))

print(mlp.score(X_test, y_test))
mlp = MLPClassifier(solver='lbfgs', random_state=3, alpha=0.1).fit(X_train, y_train)

print(mlp.score(X_train, y_train))

print(mlp.score(X_test, y_test))