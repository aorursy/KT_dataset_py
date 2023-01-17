import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print(train.shape)

print(test.shape)
train.info()       #trainデータの概要を確認
test.info()        #testデータの概要を確認
train.isnull().sum()  #trainデータの欠損値の合計を確認
test.isnull().sum()    #testデータの欠損値の合計を確認
train.head()         #trainデータの最初の5行を確認
test.head()          #testデータの最初の5行を確認
train.tail()         #trainデータの最後の5行を確認
test.tail()          #testデータの最後の5行を確認
train["Cabin"].value_counts()        #trainデータのCabinの中身を確認
train["Embarked"].value_counts()     #trainデータのEmbarkedの中身を確認
train["Ticket"].value_counts()        #trainデータのTicketの中身を確認
# Ageの穴埋め(中央値)

train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())

# Cabinは欠損値が多いため削除

train.drop("Cabin", axis=1, inplace=True)

test.drop("Cabin", axis=1, inplace=True)

# Embarkedは最頻値であるSを代入

train["Embarked"] = train["Embarked"].fillna("S")

# Fareの穴埋め(中央値)

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
train.describe()
test.describe()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
g = sns.barplot(x="Sex",y="Survived",data=train)

g = g.set_ylabel("Survival Probability")
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
print(f'[train]Nmaeのユニークな要素: {train["Name"].nunique()}')

print(f'[test]Nmaeのユニークな要素: {test["Name"].nunique()}')

print(f'[train]Sexのユニークな要素: {train["Sex"].nunique()}')

print(f'[test]Sexのユニークな要素: {test["Sex"].nunique()}')

print(f'[train]Ticketのユニークな要素: {train["Ticket"].nunique()}')

print(f'[test]Ticketのユニークな要素: {test["Ticket"].nunique()}')

print(f'[train]Embarkedのユニークな要素: {train["Embarked"].nunique()}')

print(f'[test]Embarkedのユニークな要素: {test["Embarked"].nunique()}')
# Nameは今回扱わない(苗字から家族であることを判別するようなカーネルもある)

train.drop("Name", axis=1, inplace=True)

test.drop("Name", axis=1, inplace=True)



# ticketはユニークな要素が多いため(大変だから)削除

train.drop("Ticket", axis=1, inplace=True)

test.drop("Ticket", axis=1, inplace=True)
# カテゴリカル変数(Sex,Embarked)をダミー変数で表示

train = train.join(pd.get_dummies(train["Sex"],prefix="sex"))

test = test.join(pd.get_dummies(test["Sex"],prefix="sex"))



train = train.join(pd.get_dummies(train["Embarked"],prefix="emberk"))

test = test.join(pd.get_dummies(test["Embarked"],prefix="emberk"))



# 使用後のカテゴリカル変数の削除

train.drop(["Sex", "Embarked"], axis=1, inplace=True)

test.drop(["Sex", "Embarked"], axis=1, inplace=True)
train.head()
# sklearn.ensembleの中からClassifierをインポート

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
# 目的変数と説明変数を分解する

X_train = train.drop("Survived", axis=1)   # Survivedのみ取り除く

Y_train = train["Survived"].values         # Survived値のみを表示
X_train.head()
Y_train
lr = LogisticRegression()

lr.fit(X_train, Y_train)



lr.score(X_train, Y_train)
rf_clf = RandomForestClassifier()

rf_clf = rf_clf.fit(X_train, Y_train)



rf_clf.score(X_train, Y_train)
# testデータを入れてRandomForestで予測

rf_pred = rf_clf.predict(test)
# svm = SVC()

# svm.fit(X_train, Y_train)



# svm.score(X_train, Y_train)
# AdaBoostClassifierでの学習

# ada_clf = AdaBoostClassifier()

# ada_clf = ada_clf.fit(X_train, Y_train)



# AdaBoostClassifierでの推論

# ada_pred = ada_clf.predict(test)
# bag_clf = BaggingClassifier()

# bag_clf = bag_clf.fit(X_train, Y_train)



# bag_pred = bag_clf.predict(test)
# et_clf = ExtraTreesClassifier()

# et_clf = et_clf.fit(X_train, y_train)



# et_pred = et_clf.predict(test)
# gb_clf = GradientBoostingClassifier()

# ba_clf = gb_clf.fit(X_train, y_train)



# gb_pred = gb_clf.predict(test)
rf_submit = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": rf_pred

    })

rf_submit.to_csv("rf.csv", index=False)