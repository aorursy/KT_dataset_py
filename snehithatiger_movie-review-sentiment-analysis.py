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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
train = pd.read_table('../input/train.tsv')
test = pd.read_table('../input/test.tsv')
subm = pd.read_csv('../input/sampleSubmission.csv')
train.head()
train.info()
train.describe()
test.head()
test.info()
test.describe()
subm.head()
xtrain, xtest, ytrain, ytest = train_test_split(train.Phrase, train.Sentiment, test_size = 0.2, random_state = 42)
cv = CountVectorizer(max_features = None)
cv.fit(xtrain)
xtrain_cv = cv.transform(xtrain)
xtrain_cv
xtest_cv = cv.transform(xtest)
xtest_cv
test = test.Phrase
test_cv = cv.transform(test)
test_cv
mnb = MultinomialNB(alpha = 0.5)
mnb.fit(xtrain_cv, ytrain)
pred = mnb.predict(test_cv)
pred
print('Accuracy of Naive Bayes: ', mnb.score( xtrain_cv , ytrain))
subm.Sentiment = pred
subm.to_csv("mnb_cv.csv", index = False)
subm.tail()
tv = TfidfVectorizer(max_features = None)
tv.fit(xtrain)
xtrain_tv = tv.transform(xtrain)
xtrain_tv
xtest_tv = tv.transform(xtest)
xtest_tv
test_tv = tv.transform(test)
test_tv
mnb = MultinomialNB()
mnb.fit(xtrain_tv, ytrain)
pred1 = mnb.predict(test_tv)
pred1
print('Accuracy of Naive Bayes: ', mnb.score( xtrain_tv , ytrain))
subm['Sentiment'] = pred1
subm.to_csv("mnb_tv.csv", index = False)
subm.tail()
clf = LogisticRegression(C = 1)
clf.fit(xtrain_cv, ytrain)
pred2 = clf.predict(test_cv)
pred2
print('Accuracy of Logistic Regression: ', clf.score( xtrain_cv , ytrain))
subm['Sentiment'] = pred2
subm.to_csv("lr_cv.csv", index = False)
subm.tail()
clf = LogisticRegression(C = 1)
clf.fit(xtrain_tv, ytrain)
pred3 = clf.predict(test_tv)
pred3
print('Accuracy of Logistic Regression: ', clf.score( xtrain_tv , ytrain))
subm['Sentiment'] = pred3
subm.to_csv("lr_tv.csv", index = False)
subm.tail()
from sklearn.svm import LinearSVC
svm = LinearSVC(dual = False)
svm.fit(xtrain_cv, ytrain)
pred4 = svm.predict(test_cv)
pred4
print('Accuracy of SVM: ', svm.score( xtrain_cv , ytrain))
subm['Sentiment'] = pred4
subm.to_csv("svm_cv.csv", index = False)
subm.tail()
svm = LinearSVC(dual = False)
svm.fit(xtrain_tv, ytrain)
pred5 = svm.predict(test_tv)
pred5
print('Accuracy of SVM: ', svm.score( xtrain_tv , ytrain))
subm['Sentiment'] = pred5
subm.to_csv("svm_tv.csv", index = False)
subm.tail()
etc = ExtraTreesClassifier()
etc.fit(xtrain_cv,ytrain)
pred6 = etc.predict(test_cv)
pred6
print('Accuracy of Extra Tree Classifier: ', etc.score( xtrain_cv , ytrain))
subm['Sentiment'] = pred6
subm.to_csv("etc_cv.csv", index = False)
subm.tail()
etc = ExtraTreesClassifier()
etc.fit(xtrain_tv,ytrain)
pred7 = etc.predict(test_tv)
pred7
print('Accuracy of Extra Tree Classifier: ', etc.score( xtrain_tv , ytrain))
subm['Sentiment'] = pred7
subm.to_csv("etc_tv.csv", index = False)
subm.tail()
bc = BaggingClassifier()
bc.fit(xtrain_cv,ytrain)
pred8 = bc.predict(test_cv)
pred8
print('Accuracy of Bagging Classifier: ', bc.score( xtrain_cv , ytrain))
subm['Sentiment'] = pred8
subm.to_csv("bc_cv.csv", index = False)
subm.tail()
bc = BaggingClassifier()
bc.fit(xtrain_tv,ytrain)
pred9 = bc.predict(test_tv)
pred9
print('Accuracy of Bagging Classifier: ', bc.score( xtrain_tv , ytrain))
subm['Sentiment'] = pred9
subm.to_csv("bc_tv.csv", index = False)
subm.tail()