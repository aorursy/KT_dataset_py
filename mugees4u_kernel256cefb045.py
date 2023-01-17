import numpy as np

import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names = colnames)
irisdata.head()
x = irisdata.drop('Class', axis = 1)
y = irisdata['Class']
x.head()
y.head()
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)
svcClassifier = SVC(kernel = 'poly', degree = 8)
svcClassifier.fit(x_train, y_train)
irisdata.shape
y_pred = svcClassifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred)*100)
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred))
print('classification report:\n', classification_report(y_test, y_pred))
from sklearn.svm import SVC

clf = SVC(kernel = 'rbf')

clf.fit(x_train, y_train)
y_pred1 = clf.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred1)*100)
clf1 = SVC(kernel = 'linear')

clf1.fit(x_train, y_train)
y_pred2 = clf1.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred2)*100)
clf2 = SVC(kernel = 'sigmoid')

clf2.fit(x_train, y_train)
y_pred3 = clf2.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred3)*100)
print('confusion matrix:\n', confusion_matrix(y_test, y_pred3))


print('Classificsation Report:\n', classification_report(y_test, y_pred3))
 