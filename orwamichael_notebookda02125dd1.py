import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns
dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Classification/master/DMVWrittenTests.csv')

X = dataset.iloc[:, [0, 1]].values

y = dataset.iloc[:, 2].values

dataset.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)

print('Accuracy: ', accuracy_score(y_test, y_pred))

cm
df = pd.DataFrame({'real Values':y_test, 'Predicted Values':y_pred})

df
