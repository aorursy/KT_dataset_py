import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
le = LabelEncoder()

le_test = LabelEncoder()



train["Sex"] = le.fit_transform(train["Sex"])

test["Sex"] = le_test.fit_transform(test["Sex"])



train.head()
train.isnull().sum()
train["Age"].fillna((train["Age"].mean()), inplace=True)

test["Age"].fillna((test["Age"].mean()), inplace=True)

test["Fare"].fillna((test["Fare"].mean()), inplace=True)



train
sns.heatmap(train.corr(), annot=True, linewidth=.5, cmap="YlGnBu")
X = train[["Pclass", "Sex", "Age", "Fare"]]

y = train["Survived"]



test = test[["Pclass", "Sex", "Age", "Fare"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)



cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)



classification_report(y_test, y_pred)
clf.predict(test)