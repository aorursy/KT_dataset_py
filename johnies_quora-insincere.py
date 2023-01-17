import os

import re



from scipy.sparse import csr_matrix

from scipy.sparse import hstack

import numpy as np

import pandas



from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score



import nltk

stemmer = nltk.PorterStemmer()



from collections import Counter
def fetch_data(data_path='../input'):

    

    data_train = pandas.read_csv(os.path.join(data_path, 'train.csv'))

    data_test = pandas.read_csv(os.path.join(data_path, 'test.csv'))

    

    X_train = data_train.drop(['qid', 'target'], axis=1).values

    y_train = data_train['target'].values

    X_test = data_test.drop(['qid'], axis=1).values

    

    return (X_train, y_train, X_test, data_train, data_test)
(X_train, y_train, X_test, \

            data_train, data_test) = fetch_data()
data_train.head()
class MessageToWordCounterTransform(BaseEstimator, TransformerMixin):

    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,

                 replace_urls=True, replace_numbers=True, stemming=True):

        self.strip_headers = strip_headers

        self.lower_case = lower_case

        self.remove_punctuation = remove_punctuation

        self.replace_urls = replace_urls

        self.replace_numbers = replace_numbers

        self.stemming = stemming

    

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        

        print('[INFO] Transforming Messages to Word Count')

        

        X_transformed = []

        for message in X:

            text = message[0] or ""

            if self.lower_case:

                text = text.lower()

            if self.replace_numbers:

                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)

            if self.remove_punctuation:

                text = re.sub(r'\W+', ' ', text, flags=re.M)

            word_counts = Counter(text.split())

            if self.stemming and stemmer is not None:

                stemmed_word_counts = Counter()

                for word, count in word_counts.items():

                    stemmed_word = stemmer.stem(word)

                    stemmed_word_counts[stemmed_word] += count

                word_counts = stemmed_word_counts

            X_transformed.append(word_counts)

        return np.array(X_transformed)
class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size=1000):

        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):

        total_count = Counter()

        for word_count in X:

            for word, count in word_count.items():

                total_count[word] += min(count, 10)

        most_common = total_count.most_common()[:self.vocabulary_size]

        self.most_common_ = most_common

        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}

        return self

    def transform(self, X, y=None):

        

        print('[INFO] Transforming Word Counter to Word Vector Transformer')

        

        rows = []

        cols = []

        data = []

        for row, word_count in enumerate(X):

            for word, count in word_count.items():

                rows.append(row)

                cols.append(self.vocabulary_.get(word, 0))

                data.append(count)

        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
preprocess_pipeline = Pipeline([

    ("email_to_wordcount", MessageToWordCounterTransform()),

    ("wordcount_to_vector", WordCounterToVectorTransformer()),

])



X_train_transformed = preprocess_pipeline.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



kf = KFold(n_splits=5, shuffle=True, random_state=42)



log_clf = LogisticRegression(solver="liblinear", random_state=42)

score = cross_val_score(log_clf, X_train_transformed, y_train, cv=kf, verbose=3)

score.mean()
X_test_transformed = preprocess_pipeline.transform(X_test)



log_clf = LogisticRegression(solver="liblinear", random_state=42)

log_clf.fit(X_train_transformed, y_train)



y_pred = log_clf.predict(X_test_transformed)
data_test['prediction'] = y_pred

data_test = data_test.drop('question_text', axis=1)

data_test.to_csv('submission.csv', index=False)