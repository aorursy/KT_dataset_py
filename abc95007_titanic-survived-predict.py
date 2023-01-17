# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_csv(r"../input/titanic/train.csv")

test_df =  pd.read_csv(r"../input/titanic/test.csv")

combine = [train_df, test_df]
train_df.head(10)
train_df.describe(include="all")
print(train_df.info())

print("======")

print(train_df.isnull().sum())
print("Cabin unique number: ", len(train_df["Cabin"].unique()))

print(train_df["Cabin"].unique())

print("="*20)

print("Embarked unique number: ", len(train_df["Embarked"].unique()))

print(train_df["Embarked"].unique())
#groupby 很方便把資料型態放在左側, 當整體資料的分類項目, 例如性別當分類項目, 很方便的就以男性女性進行統計, 甚至可以做兩層

items = ["Pclass", "Embarked", "SibSp","Parch"]

print("Sex_Groupby")

print(train_df[["Sex", "Survived"]].groupby(["Sex"]).mean())

print("="*20)

for item in items:

    print(item + "_Groupby")

    print(train_df[["Survived", "Sex", item]].groupby(["Sex", item]).mean().sort_values(by="Survived",ascending = False))

    print("="*20)
sns.countplot("Survived", hue="Sex", data=train_df)

train_df[["Sex","Survived"]].groupby(["Sex"]).mean()
ax1 = plt.subplot(1,2,1)

sns.countplot("Pclass", hue="Survived", data=train_df, ax=ax1)

ax2 = plt.subplot(1,2,2)

sns.countplot("Pclass", hue="Sex", data=train_df, ax=ax2)

print(train_df[["Pclass", "Survived","Sex"]].groupby(["Pclass","Sex"]).mean())

pd.crosstab(train_df["Pclass"], [train_df["Survived"], train_df["Sex"]])
plt.figure(figsize=(12, 6))

ax1 = plt.subplot(1,3,1)

sns.countplot("Embarked", hue="Survived", data=train_df, ax=ax1)

ax2 = plt.subplot(1,3,2)

sns.countplot("Embarked", hue="Sex", data=train_df, ax=ax2)

ax3 = plt.subplot(1,3,3)

train_df[["Embarked", "Pclass"]].groupby(["Embarked"]).mean().plot.bar(ax=ax3)

train_df[["Embarked", "Survived", "Sex", "Pclass", "Fare"]].groupby(["Embarked", "Sex"]).mean()
#train_df[["Embarked", "Survived","Sex", "Pclass"]].groupby(["Embarked","Sex","Pclass"]).agg([np.sum, np.mean])

train_df[["Embarked", "Survived","Sex", "Pclass"]].groupby(["Pclass","Embarked","Sex"]).mean()
for i in range(1,4):

    plt.xlim(0,90)

    plt.title("pclass" + str(i))

    sns.distplot(train_df["Age"][(train_df["Survived"]==1) & (train_df["Pclass"]==i)].dropna(), hist=10, label="Survived")

    sns.distplot(train_df["Age"][(train_df["Survived"]==0) & (train_df["Pclass"]==i)].dropna(), hist=10, label="Dead")

    plt.legend()

    plt.show()
for dataset in combine:

    dataset["Title"] = dataset["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df["Title"], [train_df["Survived"], train_df["Sex"]], margins=True)
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived',"Age"]].groupby(['Title'], as_index=False).mean()
for dataset in combine:

    dataset["Age"][(dataset["Title"]=="Master") & (dataset["Age"].isnull())] =  4.57

    dataset["Age"][(dataset["Title"]=="Miss") & (dataset["Age"].isnull())] =  21.8

    dataset["Age"] [(dataset["Title"]=="Mr") & (dataset["Age"].isnull())] =  32.3

    dataset["Age"] [(dataset["Title"]=="Mrs") & (dataset["Age"].isnull())] =  35.7

    dataset["Age"] [(dataset["Title"]=="Rare") & (dataset["Age"].isnull())] =  45.5

train_df["Age"].isnull().sum()
train_df["FareBand"] = pd.qcut(train_df["Fare"],6)

plt.figure(figsize=(24,8))

sns.factorplot("FareBand", "Survived", hue="Sex", data=train_df, size=4, aspect=3)

train_df[["FareBand", "Survived","Sex"]].groupby(["FareBand","Sex"]).mean()
for dataset in combine:

    dataset["Fare"][dataset["Fare"]<7.75] = 0

    dataset["Fare"][(dataset["Fare"]>=7.75) & (dataset["Fare"]<8.66)]=1

    dataset["Fare"][(dataset["Fare"]>=8.66) & (dataset["Fare"]<14.45)]=2    

    dataset["Fare"][(dataset["Fare"]>=14.45) & (dataset["Fare"]<26)]=3    

    dataset["Fare"][(dataset["Fare"]>=26) & (dataset["Fare"]<52.36)]=4    

    dataset["Fare"][(dataset["Fare"]>=52.36)]=5

train_df.drop("FareBand", axis=1, inplace=True)

train_df.head(10)
for dataset in combine:

    dataset["Sex"] = dataset["Sex"].map({"male":0, "female":1})
for dataset in combine:

    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] +1

train_df.groupby(["FamilySize"]).mean().sort_values(by="Survived",ascending = False)
for dataset in combine:

    dataset.drop(["Name", "Ticket", "SibSp", "Parch", "Cabin"], axis=1, inplace=True)

    dataset["Embarked"].fillna("S", inplace=True)

train_df.drop("PassengerId", axis=1, inplace=True)

print(train_df.isnull().sum())

train_df.head()
print(test_df.groupby("Fare").count())

test_df["Fare"].fillna("1.0", inplace=True)

test_df.isnull().sum().head()
for dataset in combine:

    dataset["Embarked"] = dataset["Embarked"].map({"S":0, "C":1, "Q":2})

    dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Mr":2, "Mrs":3, "Rare":4})

train_df.head()
train_df["AgeBand"] = pd.qcut(train_df["Age"],6)

plt.figure(figsize=(24,8))

sns.factorplot("AgeBand", "Survived", hue="Sex", data=train_df, size=4, aspect=3)

train_df[["AgeBand", "Survived","Sex"]].groupby(["AgeBand","Sex"]).mean()
for dataset in combine:

    dataset["Age"][dataset["Age"]<7.75] = 0

    dataset["Age"][(dataset["Age"]>=7.75) & (dataset["Age"]<8.66)]=1

    dataset["Age"][(dataset["Age"]>=8.66) & (dataset["Age"]<14.45)]=2    

    dataset["Age"][(dataset["Age"]>=14.45) & (dataset["Age"]<26)]=3    

    dataset["Age"][(dataset["Age"]>=26) & (dataset["Age"]<52.36)]=4    

    dataset["Age"][(dataset["Age"]>=52.36) & (dataset["Age"]<512.32)]=5

train_df.drop("AgeBand", axis=1, inplace=True)

train_df.head(10)
train_df = train_df.astype(int)

test_df = test_df.astype(int)


# 主要前三項相關性最高分別是 Sex, Pclass, Embarked

plt.figure(figsize=(10,8))

sns.heatmap(train_df.corr(), annot=True, linewidth=1)
test_df.shape
y = train_df["Survived"]

X = train_df.drop(["Survived"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=1)

print("X_train　size", X_train.shape)

print("X_test　size", X_test.shape)

print("y_train　size", y_train.shape)

print("y_test　size", y_test.shape)

print(X_train.columns)
items = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize']

for item in items:

    logisticRegression = LogisticRegression()

    logisticRegression.fit(X_train[[item]], y_train)

    print(item)

    print(round(logisticRegression.score(X_test[[item]], y_test) *100, 2))
train_df.head()
items = []

items.append(['Pclass', 'Sex', 'Fare'])

items.append(['Pclass', 'Sex', 'Fare', 'Age'])

items.append(['Pclass', 'Sex', 'Fare', 'Age', 'Embarked'])

items.append(['Pclass', 'Sex', 'Fare', 'Age', 'Embarked', 'Title'])

items.append(['Pclass', 'Sex', 'Fare', 'Age', 'Embarked', 'Title', 'FamilySize'])

for item in items:

    logisticRegression = LogisticRegression(solver="liblinear")

    logisticRegression.fit(X_train[item], y_train)

    print(item)

    print(round(logisticRegression.score(X_test[item], y_test) *100, 2))
logisticRegression = LogisticRegression(solver="liblinear")

#cross_val_score(logisticRegression, X,y,cv=10)

logisticRegression.fit(X_train, y_train)
result = cross_val_predict(logisticRegression, X_train, y_train)

sns.heatmap(confusion_matrix(y_train, result), annot=True, fmt="2.0f")
from sklearn.model_selection import GridSearchCV

svc = SVC()

C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

kernel=['rbf','linear']

hyper={'kernel':kernel,'C':C,'gamma':gamma}

gd = GridSearchCV(svc, param_grid=hyper, verbose=True)

gd.fit(X_train, y_train)

print(gd.best_score_)

# 列印出最佳參數

print(gd.best_estimator_)
models = []

models.append(LogisticRegression(solver="liblinear"))

models.append(RandomForestClassifier())

models.append(GaussianNB())

models.append(Perceptron())

models.append(KNeighborsClassifier(n_neighbors = 3))

models.append(SGDClassifier())

models.append(DecisionTreeClassifier())

models.append(SVC())
scores = []

for model in models:

    model.fit(X_train, y_train)

    print(round(model.score(X_test, y_test) *100, 2))

scores = np.asarray(scores)
scores = []

for model in models:

    scores.append(cross_val_score(model, X,y,cv=10))

scores = np.asarray(scores).T



plt.figure(figsize=(12,5))

ax1 = plt.subplot(1,2,1)

sns.heatmap(scores, annot=True, ax=ax1)

ax2 = plt.subplot(1,2,2)

ax2.boxplot(scores)

np.mean(scores)
submit = test_df[["PassengerId"]]

submit["Survived"] = models[6].predict(test_df.drop("PassengerId", axis=1))

submit.head()
#submit.to_csv(r'C:\Users\Lee\Desktop\submit.csv', index=False)