import pandas as pd

import numpy as np

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import KFold

from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt

%matplotlib inline

iris = pd.read_csv("../input/Iris.csv")

iris.head()

iris.dtypes
iris.head()
X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

y = iris["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

logistic_reg = LogisticRegression()

logistic_reg.fit(X_train, y_train)

y_predict = logistic_reg.predict(X_test)

y_predict
accuracy = metrics.accuracy_score(y_test, y_predict)

accuracy
k_range = range(1,31)

accuracyScore = []

for k in k_range:

    knn = KNeighborsClassifier(k)

    knn.fit(X_train, y_train)

    y_predict = knn.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_predict)

    accuracyScore.append(accuracy)
plt.plot(k_range, accuracyScore)

plt.xlabel("K value for KNeighborsClassifier")

plt.ylabel("Accuracy of KNeighborsClassifier")

plt.title("Train-Test Split Method Accuracy Graph")

plt.show()
kf = KFold(25,n_folds=5, shuffle=False)

print("{}{:^61}{}".format("Iteration", "Training Data", "Testing Data"))

for iterate, data in enumerate(kf, start=1):

    print("{:^5}{}{}".format(iterate, data[0], data[1]))
knn = KNeighborsClassifier(n_neighbors=5)

accuracy = cross_val_score(knn,X, y, cv=10, scoring="accuracy")

accuracy.mean()
k_range = range(1,31)

accuracyScore = []

for k in k_range:

    knn = KNeighborsClassifier(k)

    accuracy = cross_val_score(knn,X, y, cv=10, scoring="accuracy")

    accuracyScore.append(accuracy.mean())

plt.plot(k_range, accuracyScore)

plt.xlabel("Value of K for KNeighborsClassifier")

plt.ylabel("Accuracy of KNeighborsClassifier")

plt.title("K-Fold Method Accuracy Graph")

plt.show()
knn = KNeighborsClassifier(13)

accuracy = cross_val_score(knn,X, y, cv=10, scoring="accuracy")

accuracy.mean()