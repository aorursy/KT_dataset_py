# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.model_selection import train_test_split
# text preprossing 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

# classifiers 
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

# evaluation
from sklearn import metrics

from sklearn.base import TransformerMixin # for DenseTransformer class

import os 
import json # to load corpus from the json file 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# convert sparse matrix to dense 
class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    def fit(self, X, y=None, **fit_params):
        return self

# define experiment procedure 
def experiment(x_train, y_train, x_test, y_test, mn, mx):
    text_clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(mn, mx),
                                                           max_df=0.95,
                                                           min_df=0.05,
                                                           analyzer='char_wb')),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB()), ])
    
    print('classifier: {}'.format(text_clf_pipeline.named_steps['clf']))
    text_clf_pipeline.fit(x_train, y_train)
    print('# features: {}'.format(len(text_clf_pipeline.named_steps['vect'].get_feature_names())))
    predicted = text_clf_pipeline.predict(x_test)
    print('Accuracy = {}'.format(np.mean(predicted == y_test)))
    print('classification report \n{}'.format(metrics.classification_report(y_pred=predicted, y_true=y_test)))
    print('confusion matrix \n{}'.format(metrics.confusion_matrix(y_pred=predicted, y_true=y_test)))

# load and explore the data 
lines = open('../input/ar_arz_wiki_corpus.json').readlines()
wiki_df = pd.DataFrame(json.loads(line) for line in lines)
#print(wiki_df)

data = wiki_df['text'].astype(str)
target = wiki_df['label'].astype('category')
print('categories: {}'.format(target.cat.categories))
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.10)
print('y_train categories: {}'.format(y_train.cat.categories))
print('y_test categories: {}'.format(y_test.cat.categories))
print('y_train description: \n{}'.format(y_train.describe()))
print('y_test description: \n{}'.format(y_test.describe()))
print('y_train counts: \n{}'.format(y_train.value_counts()))
print('y_test counts: \n{}'.format(y_test.value_counts()))
# check dimintionality of data 
print('dim x_train: {}'.format(x_train.shape))
print('dim y_train: {}'.format(y_train.shape))
print('dim x_test: {}'.format(x_test.shape))
print('dim y_test: {}'.format(y_test.shape))
# 4 char grams, MultinomialNB classifier
mn = 4
mx = 4
print('ngram range {}'.format((mn, mx)))
experiment(x_train, y_train, x_test, y_test, mn, mx)
# 5 char grams, MultinomialNB classifier
mn = 4
mx = 5
print('ngram range {}'.format((mn, mx)))
experiment(x_train, y_train, x_test, y_test, mn, mx)
# 6 char grams, MultinomialNB classifier
mn = 4
mx = 6
print('ngram range {}'.format((mn, mx)))
experiment(x_train, y_train, x_test, y_test, mn, mx)
# 7 char grams, MultinomialNB classifier
mn = 4
mx = 7
print('ngram range {}'.format((mn, mx)))
experiment(x_train, y_train, x_test, y_test, mn, mx)