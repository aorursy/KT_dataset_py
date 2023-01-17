# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import nltk

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, plot_confusion_matrix

from sklearn.svm import LinearSVC, SVC, NuSVC

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

import random
rand_state = random.seed(12)
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train
X = train['text']
y = train['target']
test_x = test['text']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rand_state, shuffle = True)
X_train

count_vectorizer = CountVectorizer(stop_words='english')#, min_df = 0.05, max_df = 0.9)

count_train = count_vectorizer.fit_transform(X_train)

count_test = count_vectorizer.transform(X_test)

count_train_sub = count_vectorizer.transform(X)

count_sub = count_vectorizer.transform(test_x)

count_nb = MultinomialNB()

count_nb.fit(count_train ,y_train)

count_nb_pred = count_nb.predict(count_test)

count_nb_score = accuracy_score(y_test,count_nb_pred)

print('MultinomialNaiveBayes Count Score: ', count_nb_score)

count_nb_cm = confusion_matrix(y_test, count_nb_pred)

count_nb_cm
count_bnb = BernoulliNB()

count_bnb.fit(count_train ,y_train)

count_bnb_pred = count_bnb.predict(count_test)

count_bnb_score = accuracy_score(y_test,count_bnb_pred)

print('BernoulliNaiveBayes Count Score: ', count_bnb_score)

count_bnb_cm = confusion_matrix(y_test, count_bnb_pred)

count_bnb_cm
count_lsvc = LinearSVC()

count_lsvc.fit(count_train ,y_train)

count_lsvc_pred = count_lsvc.predict(count_test)

count_lsvc_score = accuracy_score(y_test,count_lsvc_pred)

print('LinearSVC Count Score: ', count_lsvc_score)

count_lsvc_cm = confusion_matrix(y_test, count_lsvc_pred)

count_lsvc_cm


count_svc = SVC()

count_svc.fit(count_train ,y_train)

count_svc_pred = count_svc.predict(count_test)

count_svc_score = accuracy_score(y_test,count_svc_pred)

print('SVC Count Score: ', count_svc_score)

count_svc_cm = confusion_matrix(y_test, count_svc_pred)

count_svc_cm
count_nusvc = NuSVC(0.4)

count_nusvc.fit(count_train ,y_train)

count_nusvc_pred = count_nusvc.predict(count_test)

count_nusvc_score = accuracy_score(y_test,count_nusvc_pred)

print('NuSVC Count Score: ', count_nusvc_score)

count_nusvc_cm = confusion_matrix(y_test, count_nusvc_pred)

count_nusvc_cm
#modelclf = NuSVC( random_state=42)

#parameters = {'nu':[0.4,0.5,0.6], 'kernel':[ 'poly', 'rbf', 'sigmoid'],'degree':[2,3,4]}

#clf = GridSearchCV(modelclf, parameters)

#clf.fit(count_train ,y_train)

#y_predclf = clf.predict(count_test)

#accuracy_score(y_test,y_predclf)

#clf.cv_results_

#clf.best_params_

#{'': , '': , '': }
#accuracy_score(y_test,y_predclf)

#modelclf = NuSVC( random_state=42)

#parameters = {'nu':[0.2,0.3,0.4]}

#clf = GridSearchCV(modelclf, parameters)

#clf.fit(count_train ,y_train)

#y_predclf = clf.predict(count_test)

#accuracy_score(y_test,y_predclf)

#clf.cv_results_

#clf.best_params_
#{'nu': 0.4}


count_sgd = SGDClassifier()

count_sgd.fit(count_train ,y_train)

count_sgd_pred = count_sgd.predict(count_test)

count_sgd_score = accuracy_score(y_test,count_sgd_pred)

print('SGD Count Score: ', count_sgd_score)

count_sgd_cm = confusion_matrix(y_test, count_sgd_pred)

count_sgd_cm


count_lr = LogisticRegression()

count_lr.fit(count_train ,y_train)

count_lr_pred = count_lr.predict(count_test)

count_lr_score = accuracy_score(y_test,count_lr_pred)

print('LogisticRegression Count Score: ', count_lr_score)

count_lr_cm = confusion_matrix(y_test, count_lr_pred)

count_lr_cm    
tfidf_vectorizer = TfidfVectorizer(stop_words='english')#, min_df = 0.05, max_df = 0.9)

tfidf_train = tfidf_vectorizer.fit_transform(X_train)

tfidf_test = tfidf_vectorizer.transform(X_test)

tfidf_train_sub = tfidf_vectorizer.transform(X)

tfidf_sub = tfidf_vectorizer.transform(test_x)
tfidf_nb = MultinomialNB()

tfidf_nb.fit(tfidf_train, y_train)

tfidf_nb_pred = tfidf_nb.predict(tfidf_test)

tfidf_nb_score = accuracy_score(y_test,tfidf_nb_pred)

print('MultinomialNaiveBayes Tfidf Score: ', tfidf_nb_score)

tfidf_nb_cm = confusion_matrix(y_test, tfidf_nb_pred)

tfidf_nb_cm
tfidf_svc = LinearSVC()



tfidf_svc.fit(tfidf_train, y_train)



tfidf_svc_pred = tfidf_svc.predict(tfidf_test)



tfidf_svc_score = accuracy_score(y_test,tfidf_svc_pred)



print("LinearSVC Score:   %0.3f" % tfidf_svc_score)



svc_cm = confusion_matrix(y_test, tfidf_svc_pred)



svc_cm
tfidf_svc0 = SVC()



tfidf_svc0.fit(tfidf_train, y_train)



tfidf_svc_pred0 = tfidf_svc.predict(tfidf_test)



tfidf_svc_score0 = accuracy_score(y_test,tfidf_svc_pred0)



print("SVC Score:   %0.3f" % tfidf_svc_score0)



svc_cm0 = confusion_matrix(y_test, tfidf_svc_pred0)

classification_report(y_test, tfidf_svc_pred0)

svc_cm0
tfidf_nusvc = NuSVC()



tfidf_nusvc.fit(tfidf_train, y_train)



tfidf_nusvc_pred = tfidf_nusvc.predict(tfidf_test)



tfidf_nusvc_score = accuracy_score(y_test,tfidf_nusvc_pred)



print("NuSVC Score:   %0.3f" % tfidf_nusvc_score)



nusvc_cm = confusion_matrix(y_test, tfidf_nusvc_pred)

classification_report(y_test, tfidf_nusvc_pred)

nusvc_cm


tfidf_bnb = BernoulliNB()

tfidf_bnb.fit(tfidf_train, y_train)

tfidf_bnb_pred = tfidf_bnb.predict(tfidf_test)

tfidf_bnb_score = accuracy_score(y_test,tfidf_bnb_pred)

print('BernoulliNaiveBayes Tfidf Score:  %0.3f' % tfidf_bnb_score)

tfidf_bnb_cm = confusion_matrix(y_test, tfidf_bnb_pred)

tfidf_bnb_cm


tfidf_sgd = SGDClassifier()



tfidf_sgd.fit(tfidf_train, y_train)



tfidf_sgd_pred = tfidf_sgd.predict(tfidf_test)



tfidf_sgd_score = accuracy_score(y_test,tfidf_sgd_pred)



print("SGD Score:   %0.3f" % tfidf_sgd_score)



sgd_cm = confusion_matrix(y_test, tfidf_sgd_pred)



sgd_cm


tfidf_lr = LogisticRegression()



tfidf_lr.fit(tfidf_train, y_train)



tfidf_lr_pred = tfidf_lr.predict(tfidf_test)



tfidf_lr_score = accuracy_score(y_test,tfidf_lr_pred)



print("LogisticRegression Score:   %0.3f" % tfidf_lr_score)



lr_cm = confusion_matrix(y_test, tfidf_lr_pred)



lr_cm
sample_sub=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')



sample_sub


count_nusvc.fit(count_train_sub ,y)

count_nusvc_sub = count_nusvc.predict(count_sub)



sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':count_nusvc_sub})
sub.to_csv('submission.csv',index=False)