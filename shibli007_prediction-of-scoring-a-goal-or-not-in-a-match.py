from sklearn import metrics

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import sklearn

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.cross_validation import train_test_split

from sklearn import linear_model

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn import tree
train = pd.read_csv('../input/Goal_Pred.csv')
print( train.shape )
print( train.head() )
print( train.info() )
y = train['Target']

print( y )
x = train.drop(['Name', 'Target' ], axis=1)

print( x )
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(x, y, test_size = 0.33, random_state = 15)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
lm = linear_model.SGDClassifier()

lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)

print("Mean Squared Error of SGDC Classifier: ",mse,"\n")
print (metrics.classification_report(Y_test, Y_pred))
accuracy = accuracy_score(Y_pred, Y_test)

print("Accuracy of SGDC Classifier: ",accuracy,"\n")
clf_test = GaussianNB() 

clf_test = clf_test.fit(X_train, Y_train)

pred = clf_test.predict(X_test)

print (metrics.classification_report(Y_test,pred))
accuracy = accuracy_score(pred, Y_test)

print("Accuracy of Naive Bayes Classifier: ",accuracy,"\n")
clf_test2 = tree.DecisionTreeClassifier(min_samples_split = 11) 

clf_test2 = clf_test2.fit(X_train, Y_train)

pred2 = clf_test2.predict(X_test)

print (metrics.classification_report(Y_test, pred2))
accuracy2 = accuracy_score(pred2, Y_test)

print("Accuracy of Decision Tree Classifier: ",accuracy2,"\n")
clf_test3 = svm.SVC(kernel = 'rbf', C = 1.0) 

clf_test3 = clf_test3.fit(X_train, Y_train)

pred3 = clf_test3.predict(X_test)

print (metrics.classification_report(Y_test, pred3))
accuracy3 = accuracy_score(pred3, Y_test)

print("Accuracy of Support Vector MAchine Classifier: ",accuracy3,"\n")