import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
raw_test = pd.read_csv('../input/test.csv')
raw_train = pd.read_csv('../input/train.csv')

train_text = raw_train['question_text']
train_target = raw_train['target']
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(train_text, train_target, test_size = 0.2)
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)
from sklearn.feature_extraction.text import TfidfTransformer

#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB().fit(X_train_tfidf, x_test)
#clf
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer(min_df = 10, max_df = 0.9, stop_words = 'english')),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
text_clf.fit(x_train,y_train)
text_clf.score(x_test,y_test)
predict_test = raw_test['question_text']
predicted = text_clf.predict(predict_test)
my_df=pd.DataFrame(predicted)
submission_pd = pd.concat([raw_test["qid"],pd.DataFrame(predicted)],axis=1)
submission_pd.columns = ['qid','prediction']
submission_pd.to_csv("submission.csv",index=False)
