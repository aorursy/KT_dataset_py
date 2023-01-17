import numpy as np
import pandas as pd

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline # hyper parameters, cross validation
from sklearn.neighbors import KNeighborsClassifier # classification
from sklearn.linear_model import LogisticRegression # regression
from sklearn.naive_bayes import BernoulliNB # bayse
from sklearn.naive_bayes import MultinomialNB # bayse
from sklearn.model_selection import GridSearchCV # cross validation

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix # classification
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# just unix command 'ls' for making sure files @ local environment
print(check_output(["ls", "../input"]).decode("utf8"))

pd_train = pd.read_json('../input/train.json', orient='columns')
pd_test = pd.read_json('../input/test.json', orient='columns')

# for checking data stracture @ pandas
# pd.set_option('display.max_columns', 50)
# pd.set_option("display.max_rows", 50)
# print(pd_train.describe())

# cast pandas to numpy
np_test = np.array(pd_test)
np_train = np.array(pd_train)

print(np_train.shape)
# 4040 are rows, 32 are columns



# X = np_train[:,0] # giver_username_if_known # slice a column
X = np_train[:,6] # request_text # text datas # Anyway click right side, Draft Environment, train.json.
Y = np_train[:,22] # requester_received_pizza # True or False

# ---------------- 
# Validation
# Hold-out, not K-hold
shuffle = np.random.permutation(np.arange(X.shape[0])) # generate randomized numbers as many as rows # 0 to 4039

# print(shuffle)
# max(shuffle) 4039
X, Y = X[shuffle], Y[shuffle] # 0 to 4039 # randamyzed array contents

# same data
print('data shape: ', X.shape) # 4040 rows
print('label shape:', Y.shape) # 4040 rows

# print(X) # text datas
# print(Y) # True or False

#l=len(X)
l = int(4040)

# closs validation, 
# l is not number 1, just alfabet L, l = 4040
# data -> Text datas
# labels -> True or False
train_data, train_labels = X[:int(l/2)], Y[:int(l/2)] # Slice rows to half 

# print(X) # text datas
# print(len(X)) # 4040
# print(l) # 4040
# train_data.shape # 2020
print(int((3*l)/4)) # 3030

# 2020 to 3030, slice 1010 from half line
# data -> Text datas
# labels -> True or False
dev_data, dev_labels = X[int(l/2):int((3*l)/4)], Y[int(l/2):int((3*l)/4)]

dev_data.shape, dev_labels.shape # 1010, 1010

# 3030 to 4040, slice 1010 from 3030
# data -> Text datas
# labels -> True or False
test_data, test_labels = X[int((3*l)/4):], Y[int((3*l)/4):] # 3030 to 4040, Slice 1010

# Any results you write to the current directory are saved as output.
# ----- Bayse model
vect = CountVectorizer() # Bag of Words

type(vect) # sklearn.feature_extraction.text.CountVectorizer
train_data # 2020 rows of text datas
"""
# making Bag of words from train_data
# data -> text datas
# train_data -> 2020 rows of text datas
# dev_data -> 1010 rows of text datas
"""

# toarray() changes to numpy object from CountVectorizer
data = vect.fit_transform(train_data).toarray() 
devdata = vect.transform(dev_data).toarray()

# print(vect.get_feature_names()) # For example 'accompanied', 'accomplish', 'accomplished', 'accomplishment', 'according', 'account',
# print(len((vect.get_feature_names()))) # 8768 words in Bag of words
# type(data) # numpy.ndarray
# type(devdata) # numpy.ndarray
"""
data   [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)
"""
# data.shape # (2020, 8848)
# print(data[0]) # [0 0 0 ... 0 0 0]
# print(data[1]) # [0 0 0 ... 0 0 0]




# ----------- rabels part

#Use np.where to binarize train and dev set where values above and below 0.5.
# labels -> True or False
# Boolean to Int as well as train_data to Bag of words
# train_labels # array([False, False, False, ..., True, False, True], dtype=object)
b = train_labels
trainlabels = np.where(b == True, 1, 0) # 1 means True 
bl = dev_labels
devlabels = np.where(bl == True, 1, 0) # 1 means True 

b2 = test_labels
testlabels = np.where(b2 == True, 1, 0) # 1 means True 
print('Baseline Scores...')
# Run MultinomialNB Classifier
# mnb_clf = Pipeline([('vect', CountVectorizer()), ('mnclf',MultinomialNB(alpha=0.01))])
# mnb_clf = mnb_clf.fit(train_data, trainlabels)
# pred = mnb_clf.predict(dev_data)
# score1=metrics.accuracy_score(devlabels,pred)
# print 'Naive Bayes Score:',score1
best_nb = []
alphas = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]

# print(len(alphas)) # 9
# print(range(len(alphas))) # range(0, 9)
"""
for k in range(len(alphas)):
    print(k)
0
1
2
3
4
5
6
7
8
"""
"""
for k in range(len(alphas)):
    mnb_clf = Pipeline([('vect', CountVectorizer()), ('mnclf', MultinomialNB(alpha=alphas[k]))])
    pass
"""

# print(mnb_clf)
"""
Pipeline(memory=None,
     steps=[('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)), ('mnclf', MultinomialNB(alpha=10.0, class_prior=None, fit_prior=True))])
""" 
# after for loop, so alpha value is last one of alphas 
# fit() is learning
# transform() is extraction of words feature 
# Leaning, fit() by text datas and True or False from train_data, trainlabels which are quantified
# 'vect' and 'mnclf' are just name. Those are meaningless, just for the rule of function Pipeline()
for k in range(len(alphas)):
    mnb_clf = Pipeline([('vect', CountVectorizer()), ('mnclf', MultinomialNB(alpha=alphas[k]))])
    mnb_clf = mnb_clf.fit(train_data, trainlabels) # leaning from text datas, True or False
    pred = mnb_clf.predict(dev_data) # pred is the prediction of Bayes
    # check the accurancy by comparing to Bayese prediction and actual result
    # metrics.accuracy_score(devlabels,pred)
    best_nb.append(metrics.accuracy_score(devlabels, pred))
    
bestAlphaAccuracy = max(best_nb)
bestAlphaValue = alphas[best_nb.index(bestAlphaAccuracy)]
print('Naive Bayes Baseline:')
print('Best Alpha =', bestAlphaValue, ' accuracy:', bestAlphaAccuracy)
print('')

# Run Logistic Regression classifier
# This codes dose not ues GridSearchCV...?
log_clf = Pipeline([('vect', CountVectorizer()),('lgclf', LogisticRegression(C=0.5))])
log_clf = log_clf.fit(train_data, trainlabels) 
pred = log_clf.predict(dev_data)        
score2= metrics.accuracy_score(devlabels,pred)
#print 'Logistic Regression Score:',score2

best_logit = []
#C = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
C= [0.8]
for k in range(len(C)):
    log_clf = Pipeline([('vect', CountVectorizer()),
                     ('lgclf', LogisticRegression(C=C[k]))]);
    log_clf = log_clf.fit(train_data, trainlabels)
    pred = log_clf.predict(dev_data)
    metrics.accuracy_score(devlabels,pred)
    # check the accurancy by comparing to Bayese prediction and actual result
    # metrics.accuracy_score(devlabels,pred)
    best_logit.append(metrics.accuracy_score(devlabels,pred))
    weights = log_clf.named_steps['lgclf'].coef_
bestCAccuracy = max(best_logit)
bestCValue = C[best_logit.index(bestCAccuracy)]
print('Logistic Regression Baseline:')
print('Best C =', bestCValue, ' accuracy:', bestCAccuracy)
print('')

