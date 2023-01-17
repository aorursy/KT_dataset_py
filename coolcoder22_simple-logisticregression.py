from IPython.display import HTML

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_is_fitted

from sklearn.linear_model import LogisticRegression

from scipy import sparse

import re

import string
HTML('<iframe width="800" height="400" src="https://www.youtube.com/embed/59bMh59JQDo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
text = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return text.sub(r' \1 ', s).split()

length = train_df.shape[0]

Vectorize = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )
train = Vectorize.fit_transform(train_df["comment_text"])

test = Vectorize.transform(test_df["comment_text"])
#Target

y = np.where(train_df['target'] >= 0.5, 1, 0)
class NbSvmClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, dual=False, n_jobs=1):

        self.C = C

        self.dual = dual

        self.n_jobs = n_jobs



    def predict(self, x):

        # Verify that model has been fit

        check_is_fitted(self, ['_r', '_clf'])

        return self._clf.predict(x.multiply(self._r))



    def predict_proba(self, x):

        # Verify that model has been fit

        check_is_fitted(self, ['_r', '_clf'])

        return self._clf.predict_proba(x.multiply(self._r))



    def fit(self, x, y):

        y = y

        x, y = check_X_y(x, y, accept_sparse=True)



        def pr(x, y_i, y):

            p = x[y==y_i].sum(0)

            return (p+1) / ((y==y_i).sum()+1)

        

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))

        x_nb = x.multiply(self._r)

        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)

        return self
NbSvm = NbSvmClassifier(C=1.5, dual=True, n_jobs=-1)

NbSvm.fit(train, y)
prediction=NbSvm.predict_proba(test)[:,1]
submission = pd.read_csv("../input/sample_submission.csv")

submission['prediction'] = prediction

submission.to_csv('submission.csv', index=False)