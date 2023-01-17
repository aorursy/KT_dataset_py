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
from mlxtend.classifier import StackingClassifier
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import svm, grid_search, datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
train = pd.read_csv('../input/train.tsv',sep = '\t')
test = pd.read_csv('../input/test.tsv',sep = '\t')
sub = pd.read_csv('../input/sampleSubmission.csv' , sep = ',')
test['Sentiment'] = -12345
train_test = pd.concat([train,test],ignore_index=True)
train_test['clean_phrase'] = train_test['Phrase'].map(lambda x: re.sub('[^a-zA-Z]',' ',x))
train_test['clean_phrase'] = train_test['Phrase'].map(lambda x: x.lower())
train_clean = train_test[train_test.Sentiment != -12345]
test_clean = train_test[train_test.Sentiment == -12345]
test_clean.drop(['Sentiment'], axis=1, inplace=True)
print(train_clean.shape)
print(test_clean.shape)
train_clean.head()
test_clean.head()
y_train = train_clean.Sentiment.values
X_train_clean = train_clean.clean_phrase.values
print ("y_train" + str(y_train.shape))
print ("X_train_clean"  + str(X_train_clean.shape))
vect = TfidfVectorizer(ngram_range=(1,3))
X_tfidf = vect.fit_transform(X_train_clean) # Using original phrase
X_tfidf.shape
from sklearn.model_selection import train_test_split
X_train , X_test, y_train , y_test = train_test_split(X_tfidf,y_train,test_size = 0.2)
clf1 = MultinomialNB()
clf2 = LinearSVC(multi_class='ovr')
#clf3 = AdaBoostClassifier(n_estimators = 30)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2], meta_classifier=lr)
%%time
sclf.fit(X_train,y_train)
stacked_clf = sclf.predict(X_test)
print ("Accuracy for Stacking 1 :" + str(metrics.accuracy_score(y_test, stacked_clf)))
clf1 = MultinomialNB()
clf2 = LinearSVC(multi_class='ovr')
clf3 = AdaBoostClassifier(n_estimators = 30)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
%%time
sclf.fit(X_train,y_train)
stacked_clf = sclf.predict(X_test)
print ("Accuracy for Stacking 2:" + str(metrics.accuracy_score(y_test, stacked_clf)))
clf1 = LogisticRegression()
clf2 = LinearSVC(multi_class='ovr')
clf3 = AdaBoostClassifier(n_estimators = 30)
nb = MultinomialNB()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=nb)
%%time
sclf.fit(X_train,y_train)
stacked_clf = sclf.predict(X_test)
print ("Accuracy for Stacking 3:" + str(metrics.accuracy_score(y_test, stacked_clf)))
clf1 = MultinomialNB()
clf2 = LinearSVC(multi_class='ovr')
clf3 = AdaBoostClassifier()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
%%time
sclf.fit(X_train,y_train)
stacked_clf = sclf.predict(X_test)
print ("Accuracy for Stacking 4:" + str(metrics.accuracy_score(y_test, stacked_clf)))

%%time
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred_gbc = gbc.predict(X_test)
print ("Accuracy for GBC:" + str(metrics.accuracy_score(y_test, y_pred_gbc)))
%%time
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
print ("Accuracy for RF:" + str(metrics.accuracy_score(y_test, y_pred_rf)))
%%time
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)
print ("Accuracy for RF:" + str(metrics.accuracy_score(y_test, y_pred_knn)))



