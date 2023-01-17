import os

import nltk

import numpy as np 

import pandas as pd 

print(os.listdir("../input"))

from nltk.corpus import stopwords

from sklearn.metrics import f1_score

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
display_all(train.head())
train['target'].value_counts()
train_text = train['question_text']

test_text = test['question_text']

train_target = train['target']

all_text = train_text.append(test_text)
tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(all_text)



count_vectorizer = CountVectorizer()

count_vectorizer.fit(all_text)



train_text_features_cv = count_vectorizer.transform(train_text)

test_text_features_cv = count_vectorizer.transform(test_text)



train_text_features_tf = tfidf_vectorizer.transform(train_text)

test_text_features_tf = tfidf_vectorizer.transform(test_text)
train_text.head()
kfold = KFold(n_splits = 5, shuffle = True, random_state = 2018)

test_preds = 0

oof_preds = np.zeros([train.shape[0],])



for i, (train_idx,valid_idx) in enumerate(kfold.split(train)):

    x_train, x_valid = train_text_features_tf[train_idx,:], train_text_features_tf[valid_idx,:]

    y_train, y_valid = train_target[train_idx], train_target[valid_idx]

    classifier = LogisticRegression()

    print('fitting.......')

    classifier.fit(x_train,y_train)

    print('predicting......')

    print('\n')

    oof_preds[valid_idx] = classifier.predict_proba(x_valid)[:,1]

    test_preds += 0.2*classifier.predict_proba(test_text_features_tf)[:,1]
pred_train = (oof_preds > .25).astype(np.int)

f1_score(train_target, pred_train)
submission1 = pd.DataFrame.from_dict({'qid': test['qid']})

submission1['prediction'] = (test_preds>0.25).astype(np.int)

submission1.to_csv('submission.csv', index=False)

submission1['prediction'] = (test_preds>0.25)