import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
ds = pd.read_csv('../input/advertising.csv')
ds.info()
ds.head(5)
sns.pairplot(ds)
ds.head(2)
ds['Age'].plot.hist(bins=40)
sns.jointplot(data=ds, x='Age', y='Area Income')
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ds, kind='kde', color='red')
sns.pairplot(ds, hue='Clicked on Ad')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix
ds.head(2)
y = ds['Clicked on Ad']

X = ds[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
model = LogisticRegression(random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print('\n')

print(confusion_matrix(y_test, y_pred))
# Predicting Model based on only two variables(Age, Daily Internet Usage)
y = ds['Clicked on Ad']

X = ds[['Daily Internet Usage', 'Age']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
model = LogisticRegression(random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print('\n')

print(confusion_matrix(y_test, y_pred))