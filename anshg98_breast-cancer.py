#import libraries

import numpy as np

from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#load dataset

cancer = datasets.load_breast_cancer()



#split into input and output

a = cancer.data

b = cancer.target



#split into training and testing

a_train, a_test, b_train, b_test = train_test_split(a, b, train_size=.6, random_state=1)
#create SVM model object

svc = svm.SVC()

#train classifier

svc.fit(a_train, b_train)
#print svm results

print('Accuracy = %.1f%%' % (accuracy_score(b_test, svc.predict(a_test)) * 100))
#import library

from sklearn.tree import DecisionTreeClassifier as DTC
#create decision tree model object

decision_tree = DTC()

#train classifier

decision_tree.fit(a_train, b_train)
#print decision tree results

print('Accuracy = %.1f%%' % (accuracy_score(b_test, decision_tree.predict(a_test)) * 100))
#import library

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#create linear discriminant model object

lda = LDA()

#train classifier

lda.fit(a_train, b_train)
#print linear discriminant results

print('Accuracy = %.1f%%' % (accuracy_score(b_test, lda.predict(a_test)) * 100))