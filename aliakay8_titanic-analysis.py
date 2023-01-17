# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Upload data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head(5)
train.info()
train[['Name', 'Age']].head(5)
train.loc[0:3,['Name', 'Age', 'Pclass', 'Survived']]
Survived,Total = len(train[train['Survived']== 1]),len(train)
print('Survived',Survived)
print('Total Passenger',Total)
print('Ratio',Survived/Total)

male, women = train['Sex'] == 'male', train['Sex'] == 'female'
sum(male), sum(women)
train.Sex.value_counts().plot(kind = 'barh', color = 'r')
plt.title("gender")
plt.grid()
plt.boxplot(train['Age'])
man_survived, man_total = sum(train['Survived'][male]), len(train['Survived'][male])
man_survived, man_total, man_survived/man_total
women_survived, women_total = sum(train['Survived'][women]), len(train['Survived'][women])
women_survived, women_total, women_survived/women_total
kids = train['Age'] < 18
kids_survived, kids_total = sum(train['Survived'][kids]), len(train['Survived'][kids])
kids_survived, kids_total, kids_survived/kids_total
plt.scatter(["M", "W","K"],[man_survived/man_total, women_survived/women_total,kids_survived/kids_total])
poor, rich = train.Pclass == 3, train.Pclass == 1

plt.subplot(1, 2, 1)
train[rich].Survived.value_counts().plot(kind = 'bar', color = 'g')
plt.title("Rich People")
plt.grid()

plt.subplot(1, 2, 2)
train[poor].Survived.value_counts().plot(kind = 'bar', color = 'r')
plt.title("Poor People")
plt.grid()

train[(train.Pclass == 1) & (train.Sex == 'female')].Survived.value_counts().plot(kind = 'barh', color = 'b')
plt.title("First rich and women")
plt.grid()

train[(train.Pclass == 3) & (train.Sex == 'male')].Survived.value_counts().plot(kind = 'barh', color = 'r')
plt.title("Last Poor and men")
plt.grid()
#Lost Values
print(pd.isnull(train).sum())
train['Age'] = train['Age'].fillna(np.mean(train['Age']))
#Feature Selection
train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
train.Sex=pd.get_dummies(train.Sex)
test.Sex=pd.get_dummies(test.Sex)
import seaborn as sns
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#Model Creation
X = train.drop('Survived', axis = 1)
y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix
print('Train Accuracy: ', accuracy_score(y_train, classifier.predict(X_train)))
print('test Accuracy:', accuracy_score(y_test, classifier.predict(X_test)))
#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)
print('train Accuracy:' , accuracy_score(y_train, lr.predict(X_train)))
print('test Accuracy:', accuracy_score(y_test, lr.predict(X_test)))
#confusion metices
print("Confusion Metrices: \n {0}".format(confusion_matrix(y_test, lr.predict(X_test))))
