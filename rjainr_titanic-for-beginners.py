import pandas as pd
import seaborn as sns
import numpy as np
import numpy.random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
train.head()
print(test.shape)
test.head()
len_train = len(train)
dataset = pd.concat([train, test], ignore_index=True)
print(dataset.shape)
dataset.head()
dataset.describe()
sns.barplot(x='Sex', y='Survived', data=train)
sns.barplot(x= 'Pclass', y='Survived', data=train)
sns.barplot(x='Embarked', y='Survived', data=train)
sns.violinplot(x='Survived', y= 'Age', data=train)
dataset.isnull().sum()
age_mean = dataset['Age'].mean()
age_std = dataset['Age'].std()
nan_count = dataset['Age'].isnull().sum()
dataset['Age'][np.isnan(dataset['Age'])] = rnd.randint(age_mean - age_std, age_mean + age_std, size= nan_count)
dataset = dataset.drop(['Cabin', 'Ticket'], axis =1)
dataset['Fare'][np.isnan(dataset['Fare'])] = dataset['Fare'].mean()
top = dataset['Embarked'].describe().top
dataset['Embarked'] = dataset['Embarked'].fillna(top)
dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1})
dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q':2})
print(dataset.isnull().sum())
dataset.head()
title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset['Title'] = pd.Series(title)
dataset['Title'].head()
dataset['Title'].describe()
sns.countplot(x="Title",data=dataset)
g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45) 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don',
                                             'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1,
                                         "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)
g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

dataset.head()
train = dataset[:len_train]
test = dataset[len_train:]
test.drop(['Survived'], axis=1, inplace=True)
Y_train = train['Survived'].astype(int)
X_train = train.drop(['Survived'], axis=1)
X_train.drop(labels=["PassengerId"], axis=1, inplace=True)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

X_train.head()
clf_log = LogisticRegression()
clf_log.fit(X_train, Y_train)
acc_log = round(clf_log.score(X_train, Y_train)*100, 2)
acc_log
clf_rnd = RandomForestClassifier()
clf_rnd.fit(X_train, Y_train)
acc_rnd = round(clf_rnd.score(X_train, Y_train)*100, 2)
acc_rnd
clf_svc = LinearSVC()
clf_svc.fit(X_train, Y_train)
acc_svc = round(clf_svc.score(X_train, Y_train)*100, 2)
acc_svc
clf_knc = KNeighborsClassifier()
clf_knc.fit(X_train, Y_train)
acc_knc = round(clf_knc.score(X_train, Y_train)*100, 2)
acc_knc
clf_gc = GaussianNB();
clf_gc.fit(X_train, Y_train)
acc_gc = round(clf_gc.score(X_train, Y_train)*100, 2)
acc_gc
y_rnd = clf_rnd.predict(test)


submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": y_rnd
})

submission.to_csv('submission.csv', sep=',', encoding='utf-8', index=False)
