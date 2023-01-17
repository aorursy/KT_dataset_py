import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sbn



%matplotlib inline
dt_train = pd.read_csv("../input/train.csv")

dt_test = pd.read_csv("../input/test.csv")
dt_train.head()
dt_train.info()
dt_train.describe()
#Survived = dt_train['Survived']

#dt_train.drop('Survived', axis=1, inplace=True)

dt = pd.concat([dt_train.drop('Survived', axis=1), dt_test])

dt.info()
plt.figure(figsize=(14,7))

sbn.heatmap(dt.isnull(), yticklabels=False, cbar=False)
print(dt['Embarked'].isnull().sum())

print(dt['Fare'].isnull().sum())
dt['Age'].hist()
dt_train['Age'].fillna(dt['Age'].median(skipna=True), inplace=True)

dt_test['Age'].fillna(dt['Age'].median(skipna=True), inplace=True)
dt_train['Fare'].fillna(dt['Fare'].mean(skipna=True), inplace=True)

dt_test['Fare'].fillna(dt['Fare'].mean(skipna=True), inplace=True)
print(dt['Embarked'].value_counts())

print(pd.crosstab(dt_train.Survived, dt_train.Embarked))
dt_train['Embarked'].fillna('S', inplace=True)

dt_test['Embarked'].fillna('S', inplace=True)
dt = pd.concat([dt_train.drop('Survived', axis=1), dt_test])

dt.info()
dt_train.head()
plt.figure(figsize=(12,6))

sbn.countplot(x="Sex", data=dt_train, hue="Survived", palette="Set1")

plt.title('Sex x Survived')
plt.figure(figsize=(12,6))

sbn.countplot(x="Embarked", data=dt_train, hue="Survived", palette="Set1")

plt.title('Embarked x Survived')
intervals = (0, dt['Age'].quantile(q=0.08), dt['Age'].quantile(q=0.75), 150)

cats = ["under_ages", "between_ages", "upper_ages"]



dt_train["Age_cat"] = pd.cut(dt_train.Age, intervals, labels=cats)

dt_test["Age_cat"] = pd.cut(dt_test.Age, intervals, labels=cats)



dt_train.drop('Age', axis=1, inplace=True)

dt_test.drop('Age', axis=1, inplace=True)
plt.figure(figsize=(12,6))

sbn.countplot(x="Age_cat", data=dt_train, hue="Survived", palette="Set1")

plt.title('Age_cat x Survived')
dt['Fare'].hist(bins=16)

print(dt['Fare'].quantile(q=[0.5,0.75]))
intervals = (dt['Fare'].min(), dt['Fare'].quantile(q=0.5), dt['Fare'].quantile(q=0.75), dt['Fare'].max())

cats = ["cheap", "expensive", "millionaire"]



dt_train["Fare_cat"] = pd.cut(dt_train.Fare, intervals, labels=cats)

dt_test["Fare_cat"] = pd.cut(dt_test.Fare, intervals, labels=cats)



dt_train.drop('Fare', axis=1, inplace=True)

dt_test.drop('Fare', axis=1, inplace=True)
plt.figure(figsize=(12,6))

sbn.countplot(x="Fare_cat", data=dt_train, hue="Survived", palette="Set1")

plt.title('Fare_cat x Survived')
dt_train.head()
dt_train.drop(['Name','Cabin', 'Ticket'], axis=1, inplace=True)

dt_test.drop(['Name','Cabin', 'Ticket'], axis=1, inplace=True)
dt_train['Family'] = dt_train['SibSp']+dt_train['Parch']+1

dt_test['Family'] = dt_test['SibSp']+dt_test['Parch']+1



dt_train.drop(['SibSp', 'Parch'], axis=1, inplace=True)

dt_test.drop(['SibSp', 'Parch'], axis=1, inplace=True)
print(dt_train['Family'].value_counts())
intervals = (0, 1, 2, 11)

cats = ["alone", "couple", "family"]



dt_train["Fam_cat"] = pd.cut(dt_train.Family, intervals, labels=cats)

dt_test["Fam_cat"] = pd.cut(dt_test.Family, intervals, labels=cats)



dt_train.drop('Family', axis=1, inplace=True)

dt_test.drop('Family', axis=1, inplace=True)
plt.figure(figsize=(12,6))

sbn.countplot(x="Fam_cat", data=dt_train, hue="Survived", palette="Set1")

plt.title('Fam_cat x Survived')
dt_train = pd.get_dummies(dt_train, columns=['Pclass', 'Sex', 'Embarked', 'Age_cat', 'Fare_cat', 'Fam_cat'], drop_first=True)

dt_test = pd.get_dummies(dt_test, columns=['Pclass', 'Sex', 'Embarked', 'Age_cat', 'Fare_cat', 'Fam_cat'], drop_first=True)

dt_train.head(10)
from xgboost import XGBClassifier
classifier =  XGBClassifier(n_estimators=1000, learning_rate=0.05,n_jobs=-1)

X_train = dt_train.drop(['Survived', 'PassengerId'], axis=1)

y_train = dt_train['Survived']
classifier.fit(X_train, y_train)
predictions = classifier.predict(dt_test.drop('PassengerId', axis=1))

output = pd.DataFrame()

output['PassengerId'] = dt_test['PassengerId']

output['Survived'] = predictions
output.head()
output.to_csv("output.csv", index=False)