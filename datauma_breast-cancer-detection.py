# import all the necessary files



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os



print(os.listdir("../input"))



data = pd.read_csv('../input/data.csv')
data.head()
y = data['diagnosis']

del_list = ['diagnosis', 'Unnamed: 32' ]

X = data.drop(del_list, axis=1)
ax = sns.countplot(y)

B,M = y.value_counts()

print('Number of Begign ', B)

print('Number of Malignant ', M)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train,y_test = train_test_split(X, y, random_state=84)
X_train.shape, X_test.shape, y_train.shape, y_test.shape 
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)
knn.predict(X_test)
knn.score(X_test, y_test)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



clf = LogisticRegression().fit(X_train, y_train)



print("Breast cancer detection")

print('Accuracy on training data set: {:.2f}'.format(clf.score(X_train, y_train)))

print('Accuracy on test data : {:.2f}'.format(clf.score(X_test, y_test)))