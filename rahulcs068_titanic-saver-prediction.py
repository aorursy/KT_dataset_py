import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

train.head(10)
gender = pd.read_csv("../input/gender_submission.csv")

gender.head(10)
test = pd.read_csv("../input/test.csv")

test.head(10)
print("Number of columns in dataset: ", train.shape[1])

print("Number of rows in dataset: ", train.shape[0])
print("Number of columns in dataset: ", gender.shape[1])

print("Number of rows in dataset: ", gender.shape[0])
print("Number of columns in dataset: ", test.shape[1])

print("Number of rows in dataset: ", test.shape[0])
survived = train[train['Survived'] == 1]

not_survived = train[train['Survived'] == 0]
print("Number of passenger survived: ", len(survived))

print("Number of passenger not survived: ", len(not_survived))
print("% of passenger survived: ", round((len(survived) / len(train))*100,2))

print("% of passenger not survived: ", round((len(not_survived) / len(train))*100, 2))
plt.figure(figsize=(15,5))



plt.subplot(121)

plt.title("class wise passengers")

sns.countplot(x='Pclass', data = train)



plt.subplot(122)

plt.title("classwise wise passenger - survived/not survided")

sns.countplot(x='Pclass', hue='Survived', data = train)

plt.show()

plt.figure(figsize=(15,5))



plt.subplot(121)

plt.title("Passengers - Siblings")

sns.countplot(x='SibSp', data = train)



plt.subplot(122)

plt.title("Passengers with Siblings - survived/not survided")

sns.countplot(x='SibSp', hue='Survived', data = train)

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(121)

plt.title("Passengers - Parent/Children")

sns.countplot(x='Parch', data = train)



plt.subplot(122)

plt.title("Passengers with Parent/Children - survived/not survided")

sns.countplot(x='Parch', hue='Survived', data = train)

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(121)

plt.title("Passengers - Embarked")

sns.countplot(x='Embarked', data = train)



plt.subplot(122)

plt.title("Passengers Embarked - survived/not survided")

sns.countplot(x='Embarked', hue='Survived', data = train)

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(121)

plt.title("Passengers - Sex")

sns.countplot(x='Sex', data = train)



plt.subplot(122)

plt.title("Passengers Sex - survived/not survided")

sns.countplot(x='Sex', hue='Survived', data = train)

plt.show()
plt.figure(figsize=(20,40))



plt.subplot(211)

plt.title("Passengers - Age")

sns.countplot(y='Age', data = train.sort_values('Age', ascending='True'))



plt.subplot(212)

plt.title("Passengers Age - survived/not survided")

sns.countplot(y='Age', hue='Survived', data = train.sort_values('Age', ascending='True'))

plt.show()
train['Age'].hist(bins = 40)
plt.figure(figsize=(40,20))

sns.countplot(x = 'Fare', hue = 'Survived', data=train)
sns.heatmap(train.isnull(), yticklabels = 'False', cbar=False, cmap = 'Blues')
train.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

train
sns.heatmap(train.isnull(), yticklabels = 'False', cbar=False, cmap = 'Blues')
plt.figure(figsize=(15,10))

sns.boxplot(x='Sex', y='Age', data=train)
def fill_age(data):

    age=data[0]

    sex=data[1]

    

    if pd.isnull(age):

        if sex is 'male':

            return 29

        else:

            return 25

    else:

        return age
train['Age'] = train[['Age','Sex']].apply(fill_age, axis=1)

train
sns.heatmap(train.isnull(), yticklabels = 'False', cbar=False, cmap = 'Blues')
train['Age'].hist(bins = 20)
male = pd.get_dummies(train['Sex'], drop_first = True)

male
train.drop(['Sex'], axis=1, inplace = True)

train
train = pd.concat([train, male], axis=1)

train
X_train = train.drop('Survived', axis=1).values

X_train
y_train = train['Survived'].values

y_train
test.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

test
test['Age'] = test[['Age','Sex']].apply(fill_age, axis=1)

test
male_t = pd.get_dummies(test['Sex'], drop_first = True)

male_t
test.drop(['Sex'], axis=1, inplace = True)

test
test = pd.concat([test, male_t], axis=1)

test
X_test = test.values

X_test
y_test = gender['Survived'].values

y_test
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
#X_train = X

#y_train = y
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=10)

classifier.fit(X_train, y_train)
X_test
y_predict_test = classifier.predict(X_test)

y_predict_test
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict_test)

sns.heatmap(cm, annot=True, fmt="d")
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict_test))