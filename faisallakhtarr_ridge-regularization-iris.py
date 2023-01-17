import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression,Ridge

from sklearn import metrics
from sklearn.datasets import load_iris

iris_data = load_iris()

data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

data['species'] = iris_data.target



print(data.head())
X = data.drop('species',axis=1)

Y = data['species']

print("X = ",X.head())

print("Y = ",Y.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
lr = LogisticRegression()

lr.fit(X_train, Y_train)



Y_pred = lr.predict(X_test)
print('Accuracy score: ', metrics.accuracy_score(Y_test, Y_pred))

print('Precision score: ', metrics.precision_score(Y_test, Y_pred, average='micro'))

print('Recall score: ', metrics.recall_score(Y_test, Y_pred, average='micro'))

print('F1 score: ', metrics.f1_score(Y_test, Y_pred, average='micro'))

print('Confusion Matrix :\n', metrics.confusion_matrix(Y_test, Y_pred))
rr = Ridge(alpha=0.5)

rr.fit(X_train, Y_train)



Y_predRR = rr.predict(X_test)
print('Accuracy score: ', metrics.accuracy_score(Y_test, Y_predRR))

print('Precision score: ', metrics.precision_score(Y_test, Y_predRR, average='micro'))

print('Recall score: ', metrics.recall_score(Y_test, Y_predRR, average='micro'))

print('F1 score: ', metrics.f1_score(Y_test, Y_predRR, average='micro'))

print('Confusion Matrix :\n', metrics.confusion_matrix(Y_test, Y_predRR))
train_score=lr.score(X_train, Y_train)

test_score=lr.score(X_test, Y_test)



Ridge_train_score = rr.score(X_train, Y_train)

Ridge_test_score = rr.score(X_test, Y_test)
print("Logistic regression train score:", train_score)

print("Logisitic regression test score:", test_score)

print("Ridge regression train score:", Ridge_train_score)

print("Ridge regression test score:", Ridge_test_score)