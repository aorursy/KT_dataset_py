import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#1: unreliable

#0: reliable

train=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

test=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')

test=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')

train
train[train.toxic==1]
test
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer



#data prep

test=test.fillna(' ')

train=train.fillna(' ')



#tfidf

#transformer = TfidfTransformer(smooth_idf=False)

count_vectorizer = CountVectorizer(ngram_range=(1, 1))

count_vectorizer.fit_transform(train[train.toxic==1]['comment_text'])

toxic_words=count_vectorizer.get_feature_names()

cvec=count_vectorizer.transform(test['translated'])







test_words=count_vectorizer.get_feature_names()

test_words=pd.DataFrame(test_words)

test_words['teller']=cvec.sum(axis=0).reshape(-1,1)

test_words=test_words.replace(0,np.nan).dropna()

common_words=test_words.sort_values('teller',ascending=False)[0].values
count_vectorizer = CountVectorizer(ngram_range=(1, 1),vocabulary=common_words)

train_tf=count_vectorizer.fit_transform(train['comment_text'])

test_tf=count_vectorizer.transform(test['content'])

train_tf,test_tf
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(max_iter=2000)

logreg.fit(train_tf, train['toxic'].values)

predictions = logreg.predict(test_tf)

pred=pd.DataFrame(predictions,columns=['toxic'])

pred['id']=test['id']

pred.groupby('toxic').count()
pred.to_csv('submission.csv', index=False)
