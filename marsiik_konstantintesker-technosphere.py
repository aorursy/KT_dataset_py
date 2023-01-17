import pandas as pd

import numpy as np

import nltk

from nltk.corpus import stopwords

import matplotlib.pyplot as plt



%matplotlib inline
train_df = pd.read_csv('../input/train.csv', sep=',').fillna('Nan_question')

initial_train_df = train_df.copy()

train_df.head()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
maxNumFeatures = 300



# bag of letter sequences (chars)

BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures, stop_words='english',

                                      analyzer='char', ngram_range=(1,2), 

                                      binary=True, lowercase=True)



BagOfWordsExtractor.fit(pd.concat((train_df.question1, train_df.question2)).unique())



trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(train_df.question1)

trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(train_df.question2)

y = train_df.is_duplicate.values
X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import roc_auc_score
clf = LogisticRegression(C=0.1, solver='newton-cg')
logRegAccuracy = []

logRegLogLoss = []

logRegAUC = []



stratifiedCV = StratifiedKFold(n_splits=10, random_state=2)

for it, (trainInds, validInds) in enumerate(stratifiedCV.split(X, y)):

    X_train_cv = X[trainInds,:]

    X_valid_cv = X[validInds,:]



    y_train_cv = y[trainInds]

    y_valid_cv = y[validInds]



    clf.fit(X_train_cv, y_train_cv)



    y_train_hat = clf.predict_proba(X_train_cv)[:,1]

    y_valid_hat = clf.predict_proba(X_valid_cv)[:,1]



    logRegAccuracy.append(accuracy_score(y_valid_cv, y_valid_hat > 0.5))

    logRegLogLoss.append(log_loss(y_valid_cv, y_valid_hat))

    logRegAUC.append(roc_auc_score(y_valid_cv, y_valid_hat))

    print ('%d done'%it)
plt.plot(logRegAccuracy, c='r')

plt.plot(logRegLogLoss, c='g')

plt.plot(logRegAUC, c='b')
test_df = pd.read_csv('../input/test.csv', sep=',').fillna('Nan_question')

test_df.head()
testQuestion1_BOW_rep = BagOfWordsExtractor.transform(test_df.question1)

testQuestion2_BOW_rep = BagOfWordsExtractor.transform(test_df.question2)



X_test = -(testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int)
predicted = clf.predict_proba(X_test)
predicted[:, 1]
submission = pd.DataFrame()

submission['test_id'] = test_df['test_id']

submission['is_duplicate'] = predicted[:, 1]

submission.to_csv('my_attempt.csv', index=False)