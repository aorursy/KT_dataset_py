import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import svm
from sklearn import ensemble
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# PlayerID,Name,GP,MIN,PTS,FGM,FGA,FG%,3P Made,3PA,3P%,FTM,FTA,FT%,OREB,DREB,REB,AST,STL,BLK,TOV,TARGET_5Yrs

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
X = train.values[:,2:20]
y = train.values[:,21]
y = y.astype('int')
print (X)
print (y)

test_X = test.values[:,2:20]
# test_y = test.values[:,21]
print (test_X)
# print (test_y)

# replacing nan values with column mean :
inds = np.where(pd.isnull(X))
col_mean = np.nanmean(X, axis=0)
X[inds] = np.take(col_mean, inds[1])
print (inds)
print (col_mean)
# print (pd.isnull(X[:,8]))
print (X[:,8])
# testing various classifiers to choose the best accuracy.

# linear SVM
# clf = svm.SVC()
# clf.fit(X, y)
#
# SVM with sigmoid kernel
# clf = svm.SVC(kernel='sigmoid')
# clf.fit(X, y)
#
# SVM with rbf kernel
# clf = svm.SVC(kernel='rbf')
# clf.fit(X, y)
#
# SVM with poly kernel
# clf = svm.SVC(kernel='poly')
# clf.fit(X, y)
#
# adaboost 
# clf = AdaBoostClassifier(n_estimators = 350)
# clf.fit(X, y)
# 
# random forest
# clf = RandomForestClassifier(n_estimators = 250)
# clf.fit(X, y)
# 
# decision tree
# clf = DecisionTreeClassifier()
# clf.fit(X, y)
#
# extra tree
# clf = ExtraTreesClassifier()
# clf.fit(X, y)
#
# gaussian naive bayes
# clf = GaussianNB()
# clf.fit(X, y)
#
# logistic regression
# clf = linear_model.LogisticRegression()
# clf.fit(X, y)
# 
# stochastic gradient descent
# clf = SGDClassifier(loss="squared_loss", penalty="l2")
# clf = SGDClassifier(loss="hinge", penalty="l2")
# clf.fit(X, y)
# 
# multi layer perceptron
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# test_X = scaler.transform(test_X)
# print (scaler)
# print (X)
# print (test_X)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(10,5), warm_start='True')
# clf.fit(X, y)
# 
# Gradient boosting
params = {'n_estimators': 2000, 'learning_rate': 0.008}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X, y)

cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': [clf.predict([test_X[i]])[0] for i in range(440)] }
submission = pd.DataFrame(cols)
print(submission)
submission.to_csv("submission.csv", index=False)