import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB





import os

print(os.listdir("../input"))

%matplotlib inline
train_df = pd.read_csv("/kaggle/input/train.csv")

test_df = pd.read_csv("/kaggle/input/test.csv")

survivor_df = pd.read_csv("/kaggle/input/gender_submission.csv")
train_df.count()
test_df.count()
train_df["Sex"].replace(['male', 'female'], [0,1], inplace=True)

train_df["Embarked"].replace(['C', 'Q', 'S'], [0,1,2], inplace=True)

test_df["Sex"].replace(['male', 'female'], [0,1], inplace=True)

test_df["Embarked"].replace(['C', 'Q', 'S'], [0,1,2], inplace=True)
train_df["Age"].replace(np.nan, train_df["Age"].median(), inplace=True)

test_df["Age"].replace(np.nan, test_df["Age"].median(), inplace=True)
train_df["Embarked"].replace(np.nan, train_df["Embarked"].median(), inplace=True)
train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 0) & (train_df['Age'] < 4), 0)

train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 4) & (train_df['Age'] < 18), 1)

train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 18) & (train_df['Age'] < 20), 2)

train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 20) & (train_df['Age'] < 29), 3)

train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 29) & (train_df['Age'] < 39), 4)

train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 39) & (train_df['Age'] < 49), 5)

train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 49) & (train_df['Age'] < 59), 6)

train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 59) & (train_df['Age'] < 69), 7)

train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 69) & (train_df['Age'] <= 80), 8)
test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 0) & (test_df['Age'] < 4), 0)

test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 4) & (test_df['Age'] < 18), 1)

test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 18) & (test_df['Age'] < 20), 2)

test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 20) & (test_df['Age'] < 29), 3)

test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 29) & (test_df['Age'] < 39), 4)

test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 39) & (test_df['Age'] < 49), 5)

test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 49) & (test_df['Age'] < 59), 6)

test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 59) & (test_df['Age'] < 69), 7)

test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 69) & (test_df['Age'] <= 80), 8)
train_df.info()
train_df.describe()
test_df.info()
test_df.describe()
fig, axes = plt.subplots(nrows=1, ncols=2)

fig.tight_layout()

train_df.groupby("Sex")["PassengerId"].count().plot.pie(labels=["male", "female"], ax=axes[0], title="Repartition female/male", figsize=(15,20))

train_df[train_df.Survived == 1].groupby("Sex")["PassengerId"].count().plot.pie(labels=["male", "female"], ax=axes[1], title="Repartition female/male among survivors")
fig, axes = plt.subplots(nrows=1, ncols=2)

fig.subplots_adjust(wspace=1)

train_df.groupby("Pclass")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition Pclass", figsize=(20,5))

train_df[train_df.Survived == 1].groupby("Pclass")["PassengerId"].count().plot.bar(ax=axes[1], title="Repartition Pclass among survivors")
fig, axes = plt.subplots(nrows=1, ncols=2)

fig.subplots_adjust(wspace=0.5)

train_df.groupby("Age")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition Age", figsize=(40,10))

train_df[train_df.Survived == 1].groupby("Age")["PassengerId"].count().plot.bar(x=axes[1], title="Repartition Age among survivors")
fig, axes = plt.subplots(nrows=1, ncols=2)

fig.subplots_adjust(wspace=1)

train_df.groupby("SibSp")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition SibSp", figsize=(20,5))

train_df[train_df.Survived == 1].groupby("SibSp")["PassengerId"].count().plot.bar(ax=axes[1], title="Repartition SibSp among survivors")
fig, axes = plt.subplots(nrows=1, ncols=2)

fig.subplots_adjust(wspace=1)

train_df.groupby("Parch")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition Parch", figsize=(20,5))

train_df[train_df.Survived == 1].groupby("Parch")["PassengerId"].count().plot.bar(ax=axes[1], title="Repartition Parch among survivors")
fig, axes = plt.subplots(nrows=1, ncols=2)

fig.subplots_adjust(wspace=1)

train_df.groupby("Embarked")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition Embarked", figsize=(20,5))

train_df[train_df.Survived == 1].groupby("Embarked")["PassengerId"].count().plot.bar(ax=axes[1], title="Repartition Embarked among survivors")
train_df["NbRelatives"] = train_df["Parch"] + train_df["SibSp"]

test_df["NbRelatives"] = test_df["Parch"] + test_df["SibSp"]
train_df.drop(columns=["Cabin", "Ticket", "Parch", "SibSp"], inplace=True)

test_df.drop(columns=["Cabin", "Ticket", "Parch", "SibSp"], inplace=True)
score_list = []
X_train = train_df[["NbRelatives", "Sex", "Pclass", "Embarked", "Age"]]

y_train = train_df["Survived"]



X_test = test_df[["NbRelatives", "Sex", "Pclass", "Embarked", "Age"]]
regressor = LogisticRegression()

regressor.fit(X_train, y_train)
score_list.append(regressor.score(X_train, y_train))
y_pred = regressor.predict(X_test)
corr_df = pd.DataFrame({'features':X_train.columns})

corr_df["Corr"] = pd.Series(regressor.coef_[0])

corr_df
clf = RandomForestClassifier(n_estimators=128, max_depth=2, random_state=0)

clf.fit(X_train, y_train)
score_list.append(clf.score(X_train, y_train))
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

score_list.append(clf.score(X_train, y_train))
prediction = clf.predict(X_test)
submit_df = pd.DataFrame({'PassengerId': survivor_df["PassengerId"], 'Survived': prediction})
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X_train, y_train)

score_list.append(clf.score(X_train, y_train))
gnb = GaussianNB()

gnb.fit(X_train, y_train)

score_list.append(gnb.score(X_train, y_train))
models = ["Logistic Regression", "Random Forest", "Decision Tree", "Neural Networks", "Bayesian Networks"]
pd.DataFrame({"Models":models, "Score":score_list}).sort_values("Score", ascending=False)
submit_df.to_csv('submission.csv', index=False)
ls