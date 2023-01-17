# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_validate

def averageAcc(cv_results,fold):
    average = 0
    for number in cv_results:
        average+=number
    average /= fold   
    #print("Cross-validate",fold,"folds accuracy is:",average)
    return average
#Load dataset
dataset_df=pd.read_csv('/kaggle/input/full-data/full_dataset_version1.csv')
X = dataset_df.drop(['buggy'],axis = 1)
y = dataset_df['buggy']

training_X = X.loc[:59999]
testing_X = X.loc[60000:,:]
training_y = y[:60000]
testing_y = y[60000:]

training_X = pd.DataFrame(training_X)
testing_X = pd.DataFrame(testing_X)
training_y = pd.DataFrame(training_y)
testing_y = pd.DataFrame(testing_y)
# Drpping features
new_X = X.drop(['line_del','num_dir','num_file','entropy','unique_change','developer','subsystem change','awareness'],axis = 1)
print(new_X)
# If max_iter is too low, use X_scaled as the training X
X_scaled = preprocessing.scale(training_X)
svm_clf = LinearSVC(dual=False).fit(training_X, training_y.values.ravel())
svm_acc = svm_clf.score(testing_X, testing_y)

kn_clf = KNeighborsClassifier(n_neighbors= 250,weights='uniform',algorithm='auto',n_jobs=-1).fit(training_X, training_y.values.ravel())
kn_acc= kn_clf.score(testing_X, testing_y)

clf = LogisticRegression(dual=False,random_state=0).fit(X_scaled, training_y.values.ravel())
clf_acc = clf.score(testing_X, testing_y)

sgd_clf = linear_model.SGDClassifier(max_iter = 600).fit(training_X, training_y.values.ravel())
sgd_acc = sgd_clf.score(testing_X, testing_y)

ada_clf = AdaBoostClassifier(random_state=0).fit(training_X, training_y.values.ravel())
ada_acc = ada_clf.score(testing_X, testing_y)
fold = 7
#SVM
svm_clf = LinearSVC(dual=False)
svm_results = cross_validate(svm_clf,new_X,y,cv = 7)
svm_acc = averageAcc(svm_results['test_score'],fold)
#K nearest neighbor 
kn_clf = KNeighborsClassifier(n_neighbors= 250,weights='uniform',algorithm='auto',n_jobs=-1)
kn_results = cross_validate(kn_clf,new_X,y,cv = 7)
kn_acc= averageAcc(kn_results['test_score'],fold)
#LR
clf = LogisticRegression(max_iter = 850,random_state=0)
clf_results = cross_validate(clf,new_X,y,cv = 7)
clf_acc= averageAcc(clf_results['test_score'],fold)
#SGD
sgd_clf = linear_model.SGDClassifier(max_iter = 600)
sgd_results = cross_validate(sgd_clf,new_X,y,cv = 7)
sgd_acc= averageAcc(sgd_results['test_score'],fold)
#AdaBoost
ada_clf = AdaBoostClassifier(random_state=0)
ada_results = cross_validate(ada_clf,new_X,y,cv = 7)
ada_acc= averageAcc(ada_results['test_score'],fold)

#Plot
model = ["SGD","SVC","Kn","LR","ADA"]
scores = [sgd_acc,svm_acc,kn_acc,clf_acc,ada_acc]

plt.ylabel('Accuracy')
plt.xlabel('Model Name')

plt.plot(model,scores,label = "score")
plt.show()