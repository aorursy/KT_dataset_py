# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

mnist = pd.read_csv('../input/train.csv')
print (mnist.info())
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mnist.iloc[:, 1:], mnist.iloc[:, 0], 
                                                    test_size = 0.6, random_state = 0)
plt.imshow(X_train.iloc[4000, :].values.reshape(28, 28), cmap=matplotlib.cm.binary)
print (y_train.iloc[4000])
from sklearn.linear_model import SGDClassifier
pract_five = X_train.iloc[4000, :]
sgd_clf = SGDClassifier(random_state=0)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([pract_five])
pract_score = sgd_clf.decision_function([pract_five])
print (pract_score)
from sklearn.metrics import accuracy_score
y_predict = sgd_clf.predict(X_test)
print ((y_test != y_predict).sum())
print (accuracy_score(y_test, y_predict))
mnist_test = pd.read_csv('../input/test.csv')
print (mnist_test.info())
yy_test = sgd_clf.predict(mnist_test)
mnist_submission = pd.DataFrame({'ImageId': range(1,28001), 'Label' : yy_test})
print (mnist_submission.info())
print (mnist_submission.head())
mnist_submission.to_csv('accountant_mnist1.csv', index=False)
def display_scores(scores):
    print ('Scores: ', scores)
    print ('Mean: ', scores.mean())
    print ('STD: ', scores.std())

from sklearn.model_selection import cross_val_score, cross_val_predict
sgd_scores = cross_val_score(sgd_clf, X_train, y_train,
                        scoring='accuracy', cv=3)
display_scores(sgd_scores)

from sklearn.metrics import confusion_matrix
y_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
confusion_matrix(y_train, y_pred)
from sklearn.neighbors import KNeighborsClassifier
neigh_clf = KNeighborsClassifier()
neigh_clf.fit(X_train, y_train)
neigh_scores = cross_val_score(neigh_clf, X_train, y_train, scoring='accuracy', cv=3)

display_scores(neigh_scores)

from sklearn.model_selection import GridSearchCV
parameters_grid = [{'n_neighbors': [1, 3, 5]}]
grid_search = GridSearchCV(neigh_clf, parameters_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print (grid_search.best_estimator_)
print (grid_search.cv_results_)
neigh_clf = KNeighborsClassifier(n_neighbors=1)
neigh_clf.fit(X_train, y_train)
yy_test = neigh_clf.predict(mnist_test)
mnist_submission = pd.DataFrame({'ImageId': range(1,28001), 'Label' : yy_test})
print (mnist_submission.info())
print (mnist_submission.head())
mnist_submission.to_csv('accountant_mnist2.csv', index=False)