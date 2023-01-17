import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import numpy as np

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
%matplotlib inline

sns.set()
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
survived_train = df_train.Survived
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])
data['Age'] = data.Age.fillna(data.Age.median())

data['Fare'] = data.Fare.fillna(data.Fare.median())
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data_train = data.iloc[:891]

data_test = data.iloc[891:]
X = data_train.values

test = data_test.values

y = survived_train.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LogisticRegression()

lr.fit(X, y)
Y_pred = lr.predict(X)

cm = confusion_matrix(survived_train, Y_pred)

df_cm = pd.DataFrame(cm, index = [i for i in "01"],

                  columns = [i for i in "01"])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)

pd.crosstab(survived_train, Y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
accuracy_score(survived_train, Y_pred)
Y_pred = lr.predict(test)

df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('LogReg.csv', index=False)
clf = tree.DecisionTreeClassifier(max_depth=3)

clf.fit(X, y)
Y_pred = clf.predict(X)

cm = confusion_matrix(survived_train, Y_pred)

df_cm = pd.DataFrame(cm, index = [i for i in "01"],

                  columns = [i for i in "01"])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)

pd.crosstab(survived_train, Y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
accuracy_score(survived_train, Y_pred)
Y_pred = clf.predict(test)

df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('DecisionTreeClassifier.csv', index=False)
clf1 = RandomForestClassifier(max_depth=3, random_state=1)

clf1.fit(X, y)
Y_pred = clf1.predict(X)

cm = confusion_matrix(survived_train, Y_pred)

df_cm = pd.DataFrame(cm, index = [i for i in "01"],

                  columns = [i for i in "01"])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)

pd.crosstab(survived_train, Y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
accuracy_score(survived_train, Y_pred)
Y_pred = clf1.predict(test)

df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('RandomForestClassifier.csv', index=False)