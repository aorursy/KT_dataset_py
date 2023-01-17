from sklearn.datasets import load_breast_cancer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
bcancer = load_breast_cancer()

X = bcancer.data

y = bcancer.target
logreg = LogisticRegression()

logreg.fit(X, y)

y_pred = logreg.predict(X)

acc_score_lr_y_y_pred = metrics.accuracy_score(y, y_pred)

acc_score_lr_y_y_pred
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X, y)

y_pred = knn.predict(X)

acc_score_knn_y_y_pred = metrics.accuracy_score(y, y_pred)

acc_score_knn_y_y_pred
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

acc_score_lr_tts_y_y_pred = metrics.accuracy_score(y_test, y_pred)

acc_score_lr_tts_y_y_pred
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc_score_knn_tts_y_y_pred = metrics.accuracy_score(y_test, y_pred)

acc_score_knn_tts_y_y_pred
k_range = list(range(1, 30))

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(k_range, scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Testing Accuracy')

plt.show()