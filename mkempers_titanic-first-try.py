# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df_titanic = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_titanic.head()
df_titanic.shape
df_titanic.dtypes
pd.isnull(df_titanic)
# All classes
df_titanic.Pclass.unique()
# Survival in passenger classes
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
sns.set(style="whitegrid")

sns.factorplot(x="Pclass", y="Survived", data=df_titanic[['Pclass', 'Survived']])

df_titanic['Sex']
#df_titanic['Female'] = df_titanic[[df_titanic['sex'] == 'female']]
#df_titanic['Female']
sns.factorplot(x="Sex", y="Survived", data=df_titanic[['Sex', 'Survived']])
df_titanic['Age'].unique()
df_titanic['Age'] = round(df_titanic['Age'])
df_test['Age'] = round(df_test['Age'])
plt.figure(figsize=(40,15))
sns.barplot(x="Age", y="Survived",data=df_titanic)
df_titanic_children = df_titanic[df_titanic['Age'] < 18]

f = sns.factorplot(x="Sex", y="Survived", data=df_titanic_children[['Sex', 'Survived']])
plt.subplots_adjust(top=0.9)
f.fig.suptitle('Survival under 18')
f = sns.factorplot(x="Sex", y="Survived", data=df_titanic[df_titanic['Age'] < 16][['Sex', 'Survived']])
plt.subplots_adjust(top=0.9)
f.fig.suptitle('Survival under 16')
f = sns.factorplot(x="Sex", y="Survived", data=df_titanic[df_titanic['Age'] > 60][['Sex', 'Survived']])
plt.subplots_adjust(top=0.9)
f.fig.suptitle('Survival over 60')
f = sns.factorplot(x="SibSp", y="Survived", data=df_titanic[['SibSp', 'Survived']])
plt.subplots_adjust(top=0.9)
f.fig.suptitle('Survival vs siblings/spouses')
f = sns.factorplot(x="Parch", y="Survived", data=df_titanic[['Parch', 'Survived']])
plt.subplots_adjust(top=0.9)
f.fig.suptitle('Survival vs parents/children')
df_titanic['Family'] = df_titanic['SibSp'] + df_titanic['Parch']
df_test['Family'] = df_test['SibSp'] + df_test['Parch']
f = sns.factorplot(x="Family", y="Survived", data=df_titanic[['Family', 'Survived']])
plt.subplots_adjust(top=0.9)
f.fig.suptitle('Survival vs having family aboard')
df_titanic['LotsOfFamily'] = (df_titanic['Family'] > 3).astype(int)
df_test['LotsOfFamily'] = (df_test['Family'] > 3).astype(int)
f = sns.factorplot(x="LotsOfFamily", y="Survived", data=df_titanic[['LotsOfFamily', 'Survived']])
plt.subplots_adjust(top=0.9)
f.fig.suptitle('Survival vs more than 3 family members aboard')
# Look at fare distribution for survived and not survived
plt.figure(figsize=(160,30))
fig, ax = plt.subplots()
ax.set_xlim(-10,100)
ax.set(xlabel='Fare', ylabel='Count')
sns.distplot(df_titanic['Fare'], ax=ax, color='blue', label='All');
sns.distplot(df_titanic[df_titanic['Survived'] == 0]['Fare'], ax=ax, color='red', label='Died');
sns.distplot(df_titanic[df_titanic['Survived'] == 1]['Fare'], ax=ax, label='Survived', color='green');
plt.legend()


df_titanic['Cabin'].unique()
df_titanic['HasCabin'] = df_titanic['Cabin'].isnull().astype(int)
df_test['HasCabin'] = df_test['Cabin'].isnull().astype(int)
f = sns.factorplot(x="HasCabin", y="Survived", data=df_titanic[['HasCabin', 'Survived']])
plt.subplots_adjust(top=0.9)
f.fig.suptitle('Survival vs has cabin')
f = sns.factorplot(x="Embarked", y="Survived", data=df_titanic[['Embarked', 'Survived']])
plt.subplots_adjust(top=0.9)
f.fig.suptitle('Survival vs Embarked')
df_test['Age'].fillna((df_test['Age'].mean()), inplace=True)
df_test['Fare'].fillna((df_test['Fare'].mean()), inplace=True)
df_titanic['IsKid'] = (df_titanic['Age'] < 16).astype(int)
df_titanic['IsFemale'] = (df_titanic['Sex'] == 'female').astype(int)
df_titanic_cleaned = df_titanic.drop(['Name', 'Embarked','Ticket', 'Age'
                                      ,'Cabin','Family','SibSp','Parch','Sex'
                                     ], axis=1, inplace=False)
df_titanic_cleaned = pd.get_dummies(df_titanic_cleaned, columns=['Pclass'])

df_test['IsKid'] = (df_test['Age'] < 16).astype(int)
df_test['IsFemale'] = (df_test['Sex'] == 'female').astype(int)
df_test_cleaned = df_test.drop(['Name', 'Embarked','Ticket', 'Age'
                                      ,'Cabin','Family','SibSp','Parch','Sex'
                                     ], axis=1, inplace=False)
df_test_cleaned = pd.get_dummies(df_test_cleaned, columns=['Pclass'])
df_titanic_cleaned.dropna(axis=0, how='any', inplace=True)
df_titanic_cleaned.shape
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_titanic_cleaned, test_size=0.3)
X_train = train.drop(['Survived','PassengerId'], axis=1);
y_train = train['Survived']
X_test = test.drop(['Survived','PassengerId'], axis=1)
y_test = test["Survived"]
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
tree_model = DecisionTreeClassifier(max_depth = 4)
model = tree_model.fit(X = X_train, 
                       y = y_train)
scores = cross_val_score(model, X = X_test, y = y_test, cv=10)
scores.mean()
from sklearn import svm
svc = svm.SVC(kernel='rbf', C=1000)
svc.fit(X_train, y_train) 
scores = cross_val_score(svc, X = X_test, y = y_test, cv=10)
scores.mean()
from sklearn import linear_model
logistic = linear_model.LogisticRegression(C=100)
logistic.fit(X_train, y_train)
scores = cross_val_score(logistic, X = X_test, y = y_test, cv=10)
scores.mean()
import lightgbm as lgb
gbm = lgb.LGBMClassifier(n_estimators=50, silent=True)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=False)
 
scores = cross_val_score(gbm, X = X_test, y = y_test, cv=10)
scores.mean()
ids = df_test_cleaned['PassengerId']
predictions = gbm.predict(df_test_cleaned.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission1.csv', index=False)