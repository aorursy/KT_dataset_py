import os
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

import warnings
warnings.filterwarnings('ignore')

print(os.listdir("../input"))
%matplotlib inline
data = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
total=data.append(test)
print('Training data shape: {}'.format(data.shape))
print('Test data shape: {}'.format(test.shape))
total

X, y = [], []

data['text']= data['ingredients'].map(",".join)
X=data['text'].str.split(', ', expand=True)
X=data['ingredients'].map(",".join)
lb = LabelEncoder()
y = lb.fit_transform(data.cuisine)
#X, y = np.array(X), np.array(y)
print ("total examples %s" % len(y),X)

X=X[:40000]
y=y[:40000]
# Feed a word2vec with the ingredients
w2v = gensim.models.Word2Vec(list(total.ingredients), size=350, window=10, min_count=2, iter=20)  #iter = first 10 ingredients !
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb",  OneVsRestClassifier(MultinomialNB()))])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
knn_tf = Pipeline([("tfidf_vectorizer",TfidfVectorizer(analyzer=lambda x: x,sublinear_tf=True)), ("knn", OneVsRestClassifier(KNeighborsClassifier()))])
# SVM - which is supposed to be more or less state of the art 
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="rbf"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", OneVsRestClassifier(SVC(kernel="rbf")))])
svc_tfidf2 = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", OneVsRestClassifier(SVC(kernel="linear")))])
svc2 = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", OneVsRestClassifier(SVC(kernel="linear")))])
svc_ovr = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("svc_ovr",  OneVsRestClassifier(SVC()))])
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
all_models = [
    ("mult_nb", mult_nb),
   # ("mult_nb_tfidf", mult_nb_tfidf),
    ("bern_nb", bern_nb),
    ("knn_tfidf", knn_tf),
    ("svc", svc),
    ("svc2", svc),    
    ("svc_tfidf", svc_tfidf),
    ("svc_tfidf2", svc_tfidf2),
    ("svc_ovr", svc_ovr),
]


unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])


print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
