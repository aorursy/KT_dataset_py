import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv', header='infer', index_col='PassengerId')
test = pd.read_csv('../input/test.csv', header='infer', index_col='PassengerId')
train.head()
print('****** DFs shape: ******', train.shape)
print(train.info())
print(train.describe())
train.loc[:, train.isnull().any()].head()
print(train['Embarked'].value_counts())

# We will fill the missing values in Embarked with 'S' which is the most common.
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked']  = test['Embarked'].fillna('S')
# We'll fill the missing age values by the mean for training and testing data.
train['Age'].fillna(train['Age'].median(), inplace = True)
test['Age'] = test['Age'].fillna(test['Age'].mean())
train.loc[:, train.isnull().any()].head()
survived = train['Survived'] == 1
male = train['Sex'] == 'male'
female = train['Sex'] == 'female'

print('Average age of male who survived', train[survived & male].Age.mean())
print('Average age of female who survived', train[survived & female].Age.mean())
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+).', expand=False)
pd.crosstab(train['Title'], train['Sex'])
g = sns.factorplot(x="Embarked", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
print(train[['Pclass', 'Survived']].groupby('Pclass').mean())
g = sns.factorplot(x="Pclass", y="Survived", data=train,
                   size=6, kind="bar", palette="muted")
g.set_ylabels("survival probability")
print(train[['Sex', 'Survived']].groupby('Sex').mean())

g = sns.factorplot(x="Sex", y="Survived", data=train,
                   size=6, kind="bar", palette="muted")
g.set_ylabels("survival probability")
print(train[['Embarked', 'Survived']].groupby('Embarked').mean())
g = sns.factorplot(x="Embarked", y="Survived", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
sns.violinplot(x="Pclass", y="Survived", hue="Sex", data=train, split=True,
               inner="quart", palette="Set3")
sns.despine(left=True)
sns.violinplot(x="Embarked", y="Survived", hue="Sex", data=train, split=True,
               inner="quart", palette="Set2")
sns.despine(left=True)
train.drop(['Name', 'Fare', 'Ticket', 'Cabin'], axis=1).head()
train.columns
test.drop(['Name', 'Fare', 'Ticket', 'Cabin'], axis=1).head()
# Here some mapping to encode categorical variables.

sex_mapping  = {'male': 1, 'female': 0}
embark_encode = {'C': 1, 'S': 2, 'Q': 3}
test.head()
test['Sex'] = test['Sex'].map(sex_mapping)
test['Embarked'] = test['Embarked'].map(embark_encode)
train['Sex'] = train['Sex'].map(sex_mapping)
train['Embarked'] = train['Embarked'].map(embark_encode)
train.head()
X_train = train.loc[:, ['Pclass', 'Sex', 'Age', 'Embarked']]
Y_train = train.loc[:, ['Survived']]

X_test = test.loc[:, ['Pclass', 'Sex', 'Age', 'Embarked']]

print(X_test.head())
print(X_test.info())
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)
# acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_logreg)
print(round(lr.score(X_train, Y_train) * 100, 2))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

print(round(clf.score(X_train, Y_train) * 100, 2))
X_test.reset_index(inplace=True)
X_test.columns
submission = pd.concat([pd.Series(X_test["PassengerId"]), pd.Series(y_pred)], axis=1)
submission.head()
submission.to_csv('to_submit.csv', index=False)
