# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train_tweets = list(train['text'])

test_tweets = list(test['text'])



total_stuff = train_tweets + test_tweets
#Cleaning totalstuff Dataset

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, len(total_stuff)):

    tweet = re.sub('[^a-zA-Z]', ' ', total_stuff[i])

    tweet = tweet.lower()

    tweet = tweet.split()

    ps = PorterStemmer()

    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    tweet = ' '.join(tweet)

    corpus.append(tweet)
#training

tr_corpus = []

for i in range(0, len(train)):

    tweet = re.sub('[^a-zA-Z]', ' ', train_tweets[i])

    tweet = tweet.lower()

    tweet = tweet.split()

    ps = PorterStemmer()

    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    tweet = ' '.join(tweet)

    tr_corpus.append(tweet)
#Test Set Cleaning

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ts_corpus = []

for i in range(0, len(test)):

    tweet1 = re.sub('[^a-zA-Z]', ' ', test_tweets[i])

    tweet1 = tweet1.lower()

    tweet1 = tweet1.split()

    ps = PorterStemmer()

    tweet1 = [ps.stem(word) for word in tweet1 if not word in set(stopwords.words('english'))]

    tweet1 = ' '.join(tweet1)

    ts_corpus.append(tweet1)
#  Bag of Words model for Training Dataset

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

cv.fit(corpus)

train_vector = cv.transform(tr_corpus).toarray()

test_vector = cv.transform(ts_corpus).toarray()

#train_vector = cv.fit_transform(cleantrain).toarray()
#Train Test Split for Training Set

from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=10, n_repeats=1)
train_y = train['target']
# Fitting Logistic Regression to the Training set



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score as acc

from sklearn.metrics import precision_score as ps

from sklearn.metrics import recall_score as rs

from sklearn.metrics import f1_score as f1



classifier = LogisticRegression()



acc_tr_t = []

acc_val_t = []

rs_tr_t = []

rs_val_t = []

ps_tr_t = []

ps_val_t = []

f1_tr_t = []

f1_val_t = []

c = 0

for tr_i, val_i in rkf.split(train_vector):

    print(c)

    train_x, val_x = train_vector[tr_i], train_vector[val_i]

    train_y, val_y = train_y[tr_i], train_y[val_i]

    classifier.fit(train_x, train_y)

    train_p = classifier.predict(train_x)

    val_p = classifier.predict(val_x)

    acc_tr_t.append(acc(train_y, train_p))

    acc_val_t.append(acc(val_y, val_p))

    ps_tr_t.append(ps(train_y, train_p))

    ps_val_t.append(ps(val_y, val_p))

    rs_tr_t.append(rs(train_y, train_p))

    rs_val_t.append(rs(val_y, val_p))

    f1_tr_t.append(f1(train_y, train_p))

    f1_val_t.append(f1(val_y, val_p))

    c += 1



print('Train Accuracy: ', np.mean(acc_tr_t))

print('Validation Accuracy: ', np.mean(acc_val_t))



print('Train Precision Score: ', np.mean(ps_tr_t))

print('Validation Precision Score: ', np.mean(ps_val_t))



print('Train Recall Score: ', np.mean(rs_tr_t))

print('Validation Recall Score: ', np.mean(rs_val_t))



print('Train F1 Score: ', np.mean(f1_tr_t))

print('Validation F1 Score: ', np.mean(f1_val_t))
test_p = classifier.predict(test_vector)

test_p
d = {}

d['id'] = list(test['id'])

d['target'] = list(test_p)

df = pd.DataFrame(d, columns=['id', 'target'])
df.to_csv('results.csv', index=False)