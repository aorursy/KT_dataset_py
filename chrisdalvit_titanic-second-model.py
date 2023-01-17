import numpy as np 

import pandas as pd

import seaborn as sns

from sklearn import preprocessing

from sklearn import impute



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.isnull().any()
df_train.isnull().sum()
df_train.isnull().sum() * 100 / df_train.shape[0]
sns.heatmap(df_train.isnull())
ids = df_test['PassengerId']

df_train = df_train.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

df_test = df_test.drop(['Cabin','Name', 'Ticket', 'PassengerId'], axis=1)
sns.heatmap(df_train.isnull())
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0]) 

df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode()[0]) 
df_train.isnull().sum()
from sklearn import preprocessing

sex_encoder = preprocessing.LabelEncoder()

df_train['Sex'] = sex_encoder.fit_transform(df_train['Sex'])

df_train['Embarked'] = sex_encoder.fit_transform(df_train['Embarked'])



df_test['Sex'] = sex_encoder.fit_transform(df_test['Sex'])

df_test['Embarked'] = sex_encoder.fit_transform(df_test['Embarked'])
sns.heatmap(df_train.corr())
sns.catplot(x="Pclass", y="Age", data=df_train, kind="violin")
sns.catplot(x="SibSp", y="Age", data=df_train)
sns.catplot(x="Sex", y="Age", data=df_train, kind="violin")
imp = impute.KNNImputer(n_neighbors=25)

transformed_data = imp.fit_transform(df_train[['Pclass','SibSp', 'Age']])

df_train['Age'] = transformed_data[:,2]



transformed_data = imp.fit_transform(df_test[['Pclass','SibSp', 'Age']])

df_test['Age'] = transformed_data[:,2]



df_test['Fare'] = df_test['Fare'].fillna(value=df_test['Fare'].median())
df_train.isnull().sum()
df_train['Survived'].value_counts()/df_train['Survived'].count()
y = df_train['Survived']

X = df_train.drop(['Survived'], axis=1)
from sklearn import tree

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)



clf = tree.DecisionTreeClassifier(max_depth=4,class_weight={0:0.616162, 1:0.383838})

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
y_pred = clf.predict(df_test)
sub = pd.DataFrame({'PassengerId': ids, 'Survived':y_pred })

sub.head()
sub.to_csv("submission.csv", index = False)