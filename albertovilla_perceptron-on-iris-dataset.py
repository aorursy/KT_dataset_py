from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
iris = datasets.load_iris()
print(len(iris.data))
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=8)
scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
clf = Perceptron(max_iter=100, tol=1e-4, eta0=0.1, n_jobs=-1, random_state=8)
clf.fit(X_train_std, y_train)
# Number of misclassified samples
ypred = clf.predict(X_test_std)
print('Misclassified %d' % (y_test != ypred).sum())
# Accuracy score
print('Accuracy score %.2f' % accuracy_score(y_test, ypred))
clf = Perceptron(max_iter=100, tol=1e-4, eta0=0.1, n_jobs=-1, random_state=8, penalty='l2')
clf.fit(X_train_std, y_train)

# Number of misclassified samples
ypred = clf.predict(X_test_std)
print('Misclassified %d' % (y_test != ypred).sum())

# Accuracy score
print('Accuracy score %.2f' % accuracy_score(y_test, ypred))
clf = Perceptron(max_iter=100, tol=1e-4, eta0=0.1, n_jobs=-1, random_state=8, penalty='l1')
clf.fit(X_train_std, y_train)

# Number of misclassified samples
ypred = clf.predict(X_test_std)
print('Misclassified %d' % (y_test != ypred).sum())

# Accuracy score
print('Accuracy score %.2f' % accuracy_score(y_test, ypred))