import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/mnist_train.csv')

test = pd.read_csv('../input/mnist_test.csv')
X_train = train.drop(['label'], axis=1)

y_train = train['label']



X_test = test.drop(['label'], axis=1)

y_test = test['label']
X_train = np.array(X_train)

X_test = np.array(X_test)
import matplotlib.pyplot as plt



digit = X_train[0]

digit_pixels = digit.reshape(28, 28)

plt.imshow(digit_pixels)



print(y_train[0])
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(loss='hinge', random_state=42)

sgd_clf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_predict, cross_val_score



print(cross_val_predict(sgd_clf, X_train, y_train, cv=3))

print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))

from sklearn.metrics import accuracy_score



sgd_svm_pred = sgd_clf.predict(X_test)

sgd_svm_pred
sgd_accuracy = accuracy_score(y_test, sgd_svm_pred)

sgd_accuracy
sgd_clf = SGDClassifier(loss='log', random_state=42)

sgd_clf.fit(X_train, y_train)
sgd_log_pred = sgd_clf.predict(X_test)

sgd_log_pred
sgd_log_score = accuracy_score(y_test, sgd_log_pred)

sgd_log_score
from sklearn.neighbors import KNeighborsClassifier



def knn_fit(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1):

    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

    knn.fit(X_train, y_train)

    

#     print(n_neighbors, weights)



    knn_pred = knn.predict(X_test)

    

    print('KNN Score: ', knn.score(X_test, y_test))

    print('Accuracy:', accuracy_score(y_test, knn_pred))

    

knn_fit(n_neighbors=5)
