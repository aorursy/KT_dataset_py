import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

data.head(5)
data.isnull().sum()

# >>> means there is no missing values in data
# now except column "Outcome", apply StandardScaler



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

data.head()
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
X = data.iloc[:, :-1]

Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
neighbors = list(range(1,100,2))

knn_scores = {"k": [], "score": []}



for k in neighbors:

    clf = KNeighborsClassifier(n_neighbors=k)

    clf.fit(X_train, Y_train)

    pred = clf.predict(X_test)

    acc = round(accuracy_score(Y_test, pred)*100, 2)

    

    knn_scores["k"].append(k) 

    knn_scores["score"].append(acc)



knn_scores = pd.DataFrame(knn_scores)



plt.plot(knn_scores["k"], knn_scores["score"])
knn_scores.sort_values("score", ascending=False).iloc[:1, :]
clf = KNeighborsClassifier(13)

clf.fit(X_train, Y_train)

pred = clf.predict(X_test)



accuracy_score(Y_test, pred)
neighbors = list(range(1,100,2))

knn_scores = {"k": [], "score": []}



for k in neighbors:

    clf = KNeighborsClassifier(n_neighbors=k)

    acc = cross_val_score(clf, X_train, Y_train, cv=5)

    knn_scores["k"].append(k) 

    knn_scores["score"].append(round(acc.mean()*100, 2))



knn_scores = pd.DataFrame(knn_scores)



plt.plot(knn_scores["k"], knn_scores["score"])
knn_scores.sort_values("score", ascending=False).iloc[:1, :]
clf = KNeighborsClassifier(51)

clf.fit(X_train, Y_train)

pred = clf.predict(X_test)



accuracy_score(Y_test, pred)