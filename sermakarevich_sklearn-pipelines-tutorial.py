import pandas as pd
import numpy as np
from scipy import sparse

from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
df = pd.read_csv('../input/train.csv')
x = df['comment_text'].values[:5000]
y = df['toxic'].values[:5000]
# default params
scoring='roc_auc'
cv=3
n_jobs=-1
max_features = 2500
tfidf = TfidfVectorizer(max_features=max_features)
lr = LogisticRegression()
p = Pipeline([
    ('tfidf', tfidf),
    ('lr', lr)
])

cross_val_score(estimator=p, X=x, y=y, scoring=scoring, cv=cv, n_jobs=n_jobs)
class NBFeaturer(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def preprocess_x(self, x, r):
        return x.multiply(r)
    
    def pr(self, x, y_i, y):
        p = x[y==y_i].sum(0)
        return (p+self.alpha) / ((y==y_i).sum()+self.alpha)

    def fit(self, x, y=None):
        self._r = sparse.csr_matrix(np.log(self.pr(x,1,y) / self.pr(x,0,y)))
        return self
    
    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb
tfidf = TfidfVectorizer(max_features=max_features)
lr = LogisticRegression()
nb = NBFeaturer(1)
p = Pipeline([
    ('tfidf', tfidf),
    ('nb', nb),
    ('lr', lr)
])

cross_val_score(estimator=p, X=x, y=y, scoring=scoring, cv=cv, n_jobs=n_jobs)
class Lemmatizer(BaseEstimator):
    def __init__(self):
        self.l = WordNetLemmatizer()
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x = map(lambda r:  ' '.join([self.l.lemmatize(i.lower()) for i in r.split()]), x)
        x = np.array(list(x))
        return x
lm = Lemmatizer()
tfidf = TfidfVectorizer(max_features=max_features)
lr = LogisticRegression()
nb = NBFeaturer(1)
p = Pipeline([
    ('lm', lm),
    ('tfidf', tfidf),
    ('nb', nb),
    ('lr', lr)
])

cross_val_score(estimator=p, X=x, y=y, scoring=scoring, cv=cv, n_jobs=n_jobs)
max_features = 2500
lm = Lemmatizer()
tfidf_w = TfidfVectorizer(max_features=max_features, analyzer='word')
tfidf_c = TfidfVectorizer(max_features=max_features, analyzer='char')
lr = LogisticRegression()
nb = NBFeaturer(1)
p = Pipeline([
    ('lm', lm),
    ('wc_tfidfs', 
         FeatureUnion([
            ('tfidf_w', tfidf_w), 
            ('tfidf_c', tfidf_c), 
         ])
    ),
    ('nb', nb),
    ('lr', lr)
])

cross_val_score(estimator=p, X=x, y=y, scoring=scoring, cv=cv, n_jobs=n_jobs)
param_grid = [{
    'wc_tfidfs__tfidf_w__max_features': [2500], 
    'wc_tfidfs__tfidf_c__stop_words': [2500, 5000],
    'lr__C': [3.],
}]

grid = GridSearchCV(p, cv=cv, n_jobs=n_jobs, param_grid=param_grid, scoring=scoring, 
                            return_train_score=False, verbose=1)
grid.fit(x, y)
grid.cv_results_
param_grid = [{
    'wc_tfidfs__tfidf_w__max_features': [2500, 5000, 10000], 
    'wc_tfidfs__tfidf_c__stop_words': [2500, 5000, 10000],
    'lr__C': [1., 3., 4.],
}]

grid = RandomizedSearchCV(p, cv=cv, n_jobs=n_jobs, param_distributions=param_grid[0], n_iter=1, 
                          scoring=scoring, return_train_score=False, verbose=1)
grid.fit(x, y)
grid.cv_results_