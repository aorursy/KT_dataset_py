# Inspiration 1: https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams/code
# Inspiration 2: https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import time
from scipy.sparse import hstack
from scipy.special import logit, expit

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# Functions
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


def pr(y_i, y, x):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


def get_mdl(y,x, c0 = 4):
    y = y
    r = np.log(pr(1,y,x) / pr(0,y,x))
    m = LogisticRegression(C= c0, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


def multi_roc_auc_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()
model_type = 'lrchar'
todate = time.strftime("%d%m")
# read data
train = pd.read_csv('../input/reddit_train.csv', encoding = 'latin-1').fillna(' ')
test = pd.read_csv('../input/reddit_test.csv', encoding = 'latin-1').fillna(' ')

# Tf-idf
# prepare tokenizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

# create sparse matrices
n = train.shape[0]
#vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,  min_df=3, max_df=0.9, strip_accents='unicode',
#                      use_idf=1, smooth_idf=1, sublinear_tf=1 )

word_vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word', 
    min_df = 5,
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3))
#     ,
#     max_features=250000)

word_vectorizer.fit(train['BODY'])
xtrain1 = word_vectorizer.transform(train['BODY'])
xtest1 = word_vectorizer.transform(test['BODY'])

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    min_df = 3,
    ngram_range=(1, 6))
#     ,
#     max_features=250000)

char_vectorizer.fit(train['BODY'])

xtrain2 = char_vectorizer.transform(train['BODY'])
xtest2 = char_vectorizer.transform(test['BODY'])

nfolds = 5
xseed = 29
cval = 4

# data setup
xtrain = hstack([xtrain1, xtrain2], format='csr')
xtest = hstack([xtest1,xtest2], format='csr')
ytrain = np.array(train['REMOVED'].copy())

# stratified split
skf = StratifiedKFold(n_splits= nfolds, random_state= xseed)

# storage structures for prval / prfull
predval = np.zeros((xtrain.shape[0], 1))
predfull = np.zeros((xtest.shape[0], 1))
scoremat = np.zeros((nfolds,1 ))
score_vec = np.zeros((1,1))
# fit model full
m,r = get_mdl(ytrain,xtrain, c0 = cval)
preds = m.predict_proba(xtest.multiply(r))[:,1]
submission = pd.concat([pd.DataFrame(preds), test['REMOVED']], axis=1)
submission.to_csv('submission.csv', index=False)
roc_auc_score(test['REMOVED'], preds)