# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv("../input/heart.csv")
df.head()
df.info()
sns.countplot(df['sex'])
sns.countplot(df['target'], hue=df['sex'])
sns.violinplot(x='target', y='age', data=df, hue='sex', split=True)
sns.boxplot(x='target', y='age', data=df)
sns.distplot(df[df['target'] == 0]['age'])
sns.distplot(df[df['target'] == 1]['age'],color='red')
X = df.drop('target', axis=1)
y = df['target']
category_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
X = pd.get_dummies(X, columns=category_cols, drop_first=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_test.value_counts()
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
from sklearn.metrics import classification_report
test_predictions = tree.predict(X_test)
train_predictions = tree.predict(X_train)
print("TRAIN:")
print(classification_report(y_train, train_predictions))
print("TEST:")
print(classification_report(y_test, test_predictions))
feature_importance = pd.Series(data=tree.feature_importances_, index=X_train.columns)
feature_importance.sort_values(ascending=False)
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=50)
rf_clf.fit(X_train, y_train)
test_predictions = rf_clf.predict(X_test)
train_predictions = rf_clf.predict(X_train)
print("TRAIN:")
print(classification_report(y_train, train_predictions))
print("TEST:")
print(classification_report(y_test, test_predictions))
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf_clf.fit(X_train, y_train)
test_predictions = rf_clf.predict(X_test)
train_predictions = rf_clf.predict(X_train)
print("TRAIN:")
print(classification_report(y_train, train_predictions))
print("TEST:")
print(classification_report(y_test, test_predictions))
