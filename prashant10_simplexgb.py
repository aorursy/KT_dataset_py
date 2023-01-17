import pandas as pd

import numpy as np

from sklearn.metrics import log_loss

import re,string

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

import string

from sklearn.cross_validation import train_test_split

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
train.shape,test.shape
train.drop(['id','qid1','qid2'],inplace=True,axis=1)

target = train['is_duplicate']

train.drop('is_duplicate',axis=1,inplace=True)
def similar(row):

    try:

        q1 = set(re.sub("[^\w]", " ",  row['question1'].lower()).split())

        q2 = set(re.sub("[^\w]", " ",  row['question2'].lower()).split())

        return len(q1 & q2)

    except:

        return 0
def unsimilar(row):

    try:

        q1 = set(re.sub("[^\w]", " ",  row['question1'].lower()).split())

        q2 = set(re.sub("[^\w]", " ",  row['question2'].lower()).split())

        o = q1&q2

        o1 = q1 - o

        o2 = q2 - o

        return len(o1) + len(o2)

    except:

        return 0
def diffLen(row):

    try:

        return abs(len(row['question1']) - len(row['question2']))

    except:

        return 0
def puncCounts(row):

    try:

        q1 = len([w for w in row['question1'] if w in string.punctuation])/len(row['question1'])

        q2 = len([w for w in row['question2'] if w in string.punctuation])/len(row['question2'])

        return abs(q1-q2)

    except:

        return 0

def digitDiff(row):

    try:

        d1 = len([char.isdigit() for char in row['question1']])/len(row['question1'])

        d2 = len([char.isdigit() for char in row['question2']])/len(row['question2'])

        return abs(d1-d2)

    except:

        return 0





def digit1(row):

    try:

        d1 = len([char.isdigit() for char in row['question1']]) > 0

        return 1 if d1 > 0 else 0

    except:

        return 0

def digit2(row):

    try:

        d2 = len([char.isdigit() for char in row['question2']]) > 0

        return 1 if d2 > 0 else 0

    except:

        return 0
ss = set(stopwords.words('english'))

def stopWords(row):

    try:

        q1 = set(re.sub("[^\w]", " ",  row['question1'].lower()).split())

        q2 = set(re.sub("[^\w]", " ",  row['question2'].lower()).split())

        l1 = len([i for i in q1 if i in ss])

        l2 = len([i for i in q2 if i in ss])

        return abs(l1-l2)

    except:

        return 0

train_cp = pd.DataFrame()

test_cp = pd.DataFrame()
train_cp['stopwordsDiff'] = train.apply(stopWords,axis=1,raw=True)

test_cp['stopwordsDiff'] = test.apply(stopWords,axis=1,raw=True)
train_cp['digitDiff'] = train.apply(digitDiff,axis=1,raw=True)

train_cp['digit1'] = train.apply(digit1,axis=1,raw=True)

train_cp['digit2'] = train.apply(digit2,axis=1,raw=True)



test_cp['digitDiff'] = test.apply(digitDiff,axis=1,raw=True)

test_cp['digit1'] = test.apply(digit1,axis=1,raw=True)

test_cp['digit2'] = test.apply(digit2,axis=1,raw=True)
train_cp['puncCounts'] = train.apply(puncCounts,axis=1,raw=True)



test_cp['puncCounts'] = test.apply(puncCounts,axis=1,raw=True)
train_cp['diffLen'] = train.apply(diffLen,axis=1,raw=True)



test_cp['diffLen'] = test.apply(diffLen,axis=1,raw=True)
train_cp['similar'] = train.apply(similar,axis=1,raw=True)

train_cp['unsimilar'] = train.apply(unsimilar,axis=1,raw=True)



test_cp['similar'] = test.apply(similar,axis=1,raw=True)

test_cp['unsimilar'] = test.apply(unsimilar,axis=1,raw=True)
tfIdf = TfidfVectorizer(ngram_range=(1,3),stop_words='english')

train_idf = tfIdf.fit_transform(train['question1'].astype(str)+train['question2'].astype(str))

test_idf = tfIdf.transform(test['question1'].astype(str)+test['question2'].astype(str))

n_comp = 20

svd = TruncatedSVD(n_components=n_comp, algorithm='arpack')

train_svd = pd.DataFrame(svd.fit_transform(train_idf))

test_svd = pd.DataFrame(svd.transform(test_idf))



#add this train_svd and test_svd to train and test respectively

train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]

test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]

train_cp = pd.concat([train_cp, train_svd], axis=1)

test_cp = pd.concat([test_cp,test_svd],axis=1)

train_cp.shape,test_cp.shape
x_train, x_valid, y_train, y_valid = train_test_split(train_cp, target, test_size=0.2, random_state=4242)

dtrain = xgb.DMatrix(x_train,y_train)

dtest = xgb.DMatrix(x_valid)

xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 1.0,

    'colsample_bytree': 0.7,

    'silent': 1,

    'objective':'binary:logistic',

    'eval_metric':'logloss'

}

xgbc = xgb.train(xgb_params, dtrain, num_boost_round=1000, verbose_eval=20)

xpreds = xgbc.predict(dtest)

log_loss(y_valid,xpreds)
dtrain = xgb.DMatrix(train_cp,target)

dtest = xgb.DMatrix(test_cp)

xgbc = xgb.train(xgb_params, dtrain, num_boost_round=1000, verbose_eval=20)

xpreds = xgbc.predict(dtest)
sub = pd.DataFrame()

sub['test_id'] = test['test_id']

sub['is_duplicate'] = xpreds

sub.to_csv('simple_xgb.csv', index=False)