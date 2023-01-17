import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import math

%matplotlib inline



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head(5)
train.info()
test.info()
train.describe()
train[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(["Survived"], ascending=False)

#The data shows that 74% of females survived while only 18% of males survived
train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(["Survived"], ascending=False)
train[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(["Survived"], ascending=False)
train[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(["Survived"], ascending=False)
train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(["Survived"], ascending=False)
matrix = sns.heatmap(train[["Survived", "Age", "SibSp", "Parch", "Fare"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plot = sns.FacetGrid(train, col="Survived")

plot.map(plt.hist, "Age", bins = 20)
g = sns.FacetGrid(train, col="Survived")

g.map(sns.barplot, "Sex", "Fare", ci=None)
Pclass_plot = sns.barplot(x="Pclass", y="Survived", data=train, ci=None)
Parch_plot = sns.barplot(x="Parch", y="Survived", data=train, ci=None)
Sibsp_plot = sns.barplot(x="SibSp", y="Survived", data=train, ci=None)
cabin_df = train.loc[:, ["Cabin", "Survived"]]

cabin_df.loc[train["Cabin"].isnull(), "Cabin"]= 0

cabin_df.loc[train["Cabin"].notnull(), "Cabin"] = 1

cabin_df.columns = ["hasCabin", "Survived"]

cabin_plot = sns.barplot(x="hasCabin", y="Survived", data=cabin_df, ci=None)
train.Embarked.mode()
train.loc[train.Embarked.isnull(), "Embarked"] = "S"
train.loc[train.Cabin.isnull(), "Cabin"] = 0

train.loc[train.Cabin.notnull(), "Cabin"] = 1

test.loc[test.Cabin.isnull(), "Cabin"] = 0

test.loc[test.Cabin.notnull(), "Cabin"] = 1
test.loc[test.Fare.isnull(), "Fare"]= test.Fare.median()
train.FareBand = pd.qcut(train.Fare, 4)

train.FareBand.value_counts()
train.loc[(train.Fare > 0) & (train.Fare <= 7.91), "Fare"] = 0

train.loc[(train.Fare > 7.91) & (train.Fare <= 14.454), "Fare"] = 1

train.loc[(train.Fare > 14.454) & (train.Fare <= 31), "Fare"] = 2

train.loc[(train.Fare > 31) & (train.Fare <= 512.329), "Fare"] = 3



test.loc[(test.Fare > 0) & (test.Fare <= 7.91), "Fare"] = 0

test.loc[(test.Fare > 7.91) & (test.Fare <= 14.454), "Fare"] = 1

test.loc[(test.Fare > 14.454) & (test.Fare <= 31), "Fare"] = 2

test.loc[(test.Fare > 31) & (test.Fare <= 512.329), "Fare"] = 3
train.head(5)
test.head(5)
train.Sex = train.Sex.map({"female" : 0, "male" : 1}).astype(int)

test.Sex = test.Sex.map({"female" : 0, "male" : 1}).astype(int)

train.head(5)
age_matrix = np.zeros((2,3))
for i in range(2):

    for j in range(3):

        ages = train.loc[(train.Sex == i) & (train.Pclass == j + 1), "Age"].dropna()

        age_matrix[i][j] = ages.median()

print(age_matrix)
for i in range(2):

    for j in range(3):

        train.loc[(train.Age.isnull()) & (train.Sex == i) & (train.Pclass == j+1), "Age"] = age_matrix[i][j]

        test.loc[(test.Age.isnull()) & (test.Sex == i) & (test.Pclass == j+1), "Age"] = age_matrix[i][j]       
train.Embarked = train.Embarked.map({"S":0, "C":1, "Q":2}).astype(int)

test.Embarked = test.Embarked.map({"S":0, "C":1, "Q":2}).astype(int)
train = train.drop(["Name", "Ticket", "PassengerId"], axis=1)

test = test.drop(["Name", "Ticket", "PassengerId"], axis=1)
train.head(5)
test.head(5)
train.info()
test.info()
train.columns

X = train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]]

Y = train.Survived

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8)
log_reg = LogisticRegression()

log_reg.fit(X_train, Y_train)

Y_pred = log_reg.predict(X_val)

acc_log = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print("Accuracy of logistic regression: %s" % acc_log + "%")
perc = Perceptron()

perc.fit(X_train, Y_train)

Y_pred = perc.predict(X_val)

acc_perc = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print("Accuracy of perceptron algorithm: %s" % acc_perc + "%")
svm = SVC()

svm.fit(X_train, Y_train)

Y_pred = svm.predict(X_val)

acc_SVM = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print("Accuracy of SVM %s" % acc_SVM + "%")
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_val)

acc_tree = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print("Accuracy of decision tree: %s" % acc_tree + "%")
random_forest = RandomForestClassifier()

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_val)

acc_forest = round(accuracy_score(Y_pred, Y_val)*100, 2)

print("Accuracy of random forest: %s" % acc_forest + "%")
adaboost = AdaBoostClassifier()

adaboost.fit(X_train, Y_train)

Y_pred = adaboost.predict(X_val)

acc_adaboost = round(accuracy_score(Y_pred, Y_val)*100, 2)

print("Accuracy of AdaBoost: %s" % acc_adaboost + "%")
knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_val)

acc_knn = round(accuracy_score(Y_pred, Y_val) * 100, 2)

print("Accuracy of KNN: %s" % acc_knn + "%")
accuracy_df = pd.DataFrame({"Model": ["Logistic Regression", "SVM", "Perceptron", "Decision Tree", "Random Forest", "AdaBoost", "KNN"], "Score": [acc_log, acc_SVM, acc_perc, acc_tree, acc_forest, acc_adaboost, acc_knn]})

accuracy_df.sort_values(by="Score", ascending=False)
predict = adaboost.predict(test)

test = pd.read_csv("../input/test.csv")

output = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": predict})

output.to_csv("submission.csv", index=False)