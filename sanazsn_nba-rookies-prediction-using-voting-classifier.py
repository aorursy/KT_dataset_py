# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.compat import pandas as pd
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm, preprocessing

from sklearn.decomposition import PCA
test = pd.read_csv("../input/test.csv")
test_x = test.iloc[:, 2:].values
imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imp=imp.fit(test_x)
test_x = imp.transform(test_x)
dataset = pd.read_csv("../input/train.csv")
print(dataset.info())
X = dataset.iloc[:, 2:-1].values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
input_dim = X.shape[1]
imputer = imputer.fit(X)
X = imputer.transform(X)
y = dataset.iloc[:, 21].values

#Standard
sc = StandardScaler()
X_train = sc.fit_transform(X)
x_test = sc.transform(test_x)
#//////////////////////

#quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
#Xprime = quantile_transformer.fit_transform(X)
#txprime = quantile_transformer.transform(test_x)
#///////////////
Xtrain = preprocessing.normalize(X_train, norm='l2')
Xtest = preprocessing.normalize(x_test, norm='l2')
#/////////////
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(2)
# Xprime = poly.fit_transform(X)
# tsprime = poly.transform(test_x)
#/////////////////
# pca = PCA(n_components=2)
# Xprime=pca.fit_transform(Xprime)
# tsprime = pca.fit_transform(test_x)
# print(Xprime.shape)
# print(tsprime)
#////////////////////

print(X)
print(y)
print(test_x)
#Classifiers
#Choosing best one
#/////////////////////////
# model = Sequential()
# model.add(Dense(20, input_dim=input_dim))
# model.add(Activation('relu'))
# model.add(Dropout(0.15))
# model.add(Dense(10))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(Dense(1))
# model.add(Activation('softmax'))
#///////////////
#SVM-RBF
 #from sklearn.svm import SVC
 #classifier = SVC(kernel = 'rbf')
 #classifier.fit(X, y)
#
 #y_predsvm = classifier.predict(test_x)
#SVC//////////////////

 #clf = SVC()
 #clf.fit(X,y)
 #y_predsvc= clf.predict(test_x)
# preds = model.predict_classes(test_x, verbose=0)
#////////////////////
#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, y)
#print("gnb:", gnb.score(X,y))
#y_predgnb = gnb.predict(test_x)
#
# model.compile(optimizer='rmsprop', loss='mae')
#
#
# model.fit(X, y, epochs=10)
#//////////////////////////
#KNN
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X, y)
#y_predknn = knn.predict(test_x)
#print("knn:",knn.score(X,y))
#/////////////////
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
rf.fit(X,y)
#y_predrf = rf.predict(test_x)
#print("rf:",rf.score(X,y))
#/////////////////////
#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gb.fit(X, y)
#y_predgb = gb.predict(test_x)
#print("gb:", gb.score(X,y))
#////////////////
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X, y)
#y_predlr = lr.predict(test_x)
#print("lr:",lr.score(X,y))
#y_pred = gb.predict(test_x)
#y_pred = classifier.predict(test_x)
#//////////QDA
qda= QuadraticDiscriminantAnalysis()
qda.fit(X,y)
#y_predqda = qda.predict(test_x)
#///////
#SVM
#svm = svm.SVC(kernel='linear', C = 1.0)
#svm.fit(X,y)
#y_predsvm = svm.predict(test_x)
#print("svm",svm.score(X,y))
# print(clf.predict(test_x))

#/////////////
#Adaboost
adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)

adb.fit(X,y)
#y_predadb = adb.predict(test_x)
#print("adb:",adb.score(X,y))
#///////////
# Voting Classifier(LR, RF,AdaBoost,SVM,GNB,Knn,GBC)

clf1 = LogisticRegression(random_state=100)
clf2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
clf3 = SVC(gamma=2, C=1)
clf4 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
#clf3 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
clf5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
clf6 = GaussianNB()
clf7 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#clf8 = svm
#clf9 = QuadraticDiscriminantAnalysis()
# Majority Vote
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svm', clf3), ('gbc', clf4),('adb',clf5),('gnb',clf6),('knn',clf7)], voting='hard')
eclf1 = eclf1.fit(X, y)
preds= eclf1.predict(test_x)

print(preds)

cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': [eclf1.predict([test_x[i]])[0] for i in range(440)] }
submission = pd.DataFrame(cols)
print(submission)
submission.to_csv("submission1.csv", index=False)