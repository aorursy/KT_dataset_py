import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from pandas.api.types import is_string_dtype, is_numeric_dtype

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from IPython.display import Image

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction import stop_words

from sklearn.metrics import log_loss

import re

import string
df_raw = pd.read_csv('../input/train.csv', low_memory=False)
df_raw = df_raw.dropna()
df_raw.shape
df_raw[:5]
def tokenizer(s): 

    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    words = re_tok.sub(r' \1 ', s).split()

    words = [w.lower() for w in words]

    words = [w.strip() for w in words]

    words = [w for w in words if len(w) >= 3]

    words = [w for w in words if w not in stop_words.ENGLISH_STOP_WORDS]

    return words
veczr = CountVectorizer(tokenizer=tokenizer, binary=True,ngram_range=(1,3))
# drop un-used columns

df_raw = df_raw.drop(["id","qid1","qid2"],axis=1)
# train-validation split

X_train, X_val, y_train, y_val = train_test_split(df_raw.drop("is_duplicate",axis=1), df_raw["is_duplicate"], \

                                                  test_size = 0.2, random_state = 99)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

y_val = y_val.apply(lambda x: x).tolist()

y_train = y_train.apply(lambda x: x).tolist()
# Based on train set, combine q1 + q2, and build a term-doc matrix

train_list1 = X_train['question1'].apply(lambda x: x).tolist()

train_list2 = X_train['question2'].apply(lambda x: x).tolist()

train_list = []

for i in range(len(train_list1)):

    train_list.append(train_list1[i] + " " + train_list2[i])
# fit a term-doc matrix, which will be used later

trn_term_doc = veczr.fit_transform(train_list)
# transform training data, based on question1 and question2

train_term_doc1 = veczr.transform(X_train['question1'].apply(lambda x: x).tolist())

train_term_doc2 = veczr.transform(X_train['question2'].apply(lambda x: x).tolist())
# if the word doesn't exist: 0

# if the word appears in one question: 1

# if the word appears in both questions: 2 

x = train_term_doc1 + train_term_doc2

y = y_train
val_term_doc1 = veczr.transform(X_val['question1'].apply(lambda x: x).tolist())

val_term_doc2 = veczr.transform(X_val['question2'].apply(lambda x: x).tolist())
m = LogisticRegression(C=0.4, dual=True)

m.fit(x, y)

preds_train = m.predict(train_term_doc1 + train_term_doc2)

preds_prob_train = m.predict_proba(train_term_doc1 + train_term_doc2)

preds_val = m.predict(val_term_doc1 + val_term_doc2)

preds_prob_val = m.predict_proba(val_term_doc1 + val_term_doc2)

print("Accuracy of training : ", (preds_train == y_train).mean())

print("Log-loss of training : ", log_loss(y_train, preds_prob_train))

print("Accuracy of validation : ", (preds_val == y_val).mean())

print("Log-loss of validation : ", log_loss(y_val, preds_prob_val))