import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.collocations import *
import scipy

def LoadFiles(training_filename, description_filename):
    df = pd.read_csv(training_filename, encoding="ISO-8859-1")
    #samples limited to reduce runtime - but this impacts quality of feature extraction
    df = df.sample(5000)

    df["search_term"] = df["search_term"].str.lower()
    df["product_title"] = df["product_title"].str.lower()

    descr = pd.read_csv(description_filename,encoding="ISO-8859-1")
    descr["product_description"] = descr["product_description"].str.lower()
    df = df.merge(descr, on="product_uid")
    df = df.assign(prod_complete = lambda x: (x['product_title'] + ' ' + x['product_description']))
    return df   

#T
class cross_ref:

    def __init__(self):
        self.df = pd.DataFrame()
        
    def fit(self, X,y):  
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer
        import sklearn
        self.vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=1, max_df = 0.2)
        search_counts = self.vectorizer.fit_transform(X["search_term"])
        distinct_title_counts = self.vectorizer.transform(X["product_title"].drop_duplicates())
        distinct_descr_counts = self.vectorizer.transform(
            X["product_description"].drop_duplicates())
        feature_counts = scipy.sparse.vstack(
            [search_counts,distinct_title_counts,distinct_descr_counts])
        self.tfidf = TfidfTransformer()
        self.tfidf.fit(feature_counts)
        from sklearn.pipeline import make_pipeline
        self.pipeline = make_pipeline(self.vectorizer, self.tfidf)
        self.search_vecs = self.pipeline.transform(X["search_term"])
        self.title_vecs = self.pipeline.transform(X["prod_complete"])
        self.common_vecs = self.search_vecs.multiply(self.title_vecs)

        from sklearn import linear_model
        from sklearn import cross_validation
        clf = linear_model.Ridge (alpha = 0.5)
        self.fitted_clf = clf.fit(self.common_vecs,y)


    def transform(self,X):
        return self.fitted_clf(X)

df = LoadFiles('../input/train.csv','../input/product_descriptions.csv')
y=df['relevance']

cr = cross_ref()
cr.fit(df,y)
cr.common_vecs = cr.search_vecs.multiply(cr.title_vecs)
cr.common_vecs


df['sums'] = cr.common_vecs.sum(axis=1)
df['counts'] = cr.common_vecs.getnnz(axis=1)
df['mean'] = df['sums'] / df['counts']


df[df['relevance'] < 1.5].sample(3)
df[df['relevance'] > 2.5].sample(3)
notrelevant = df[df['relevance'] < 1.5]
relevant = df[df['relevance'] > 2.5]
notrelevant['sums'].mean()
relevant['sums'].mean()

import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(df['relevance'],df['sums'])
plt.scatter(df['relevance'],df['counts'])
plt.scatter(df['relevance'],df['maxes'])

#whats in bottom right of these
df[(df['relevance'] > 2.5) & (df['sums']<0.1)].sample(10)
srch_sum = cr.search_vecs.sum(axis=1)
df['srch_sum'] = srch_sum
plt.scatter(df['relevance'],df['sums']/df['srch_sum'])
plt.scatter(df['relevance'], df['mean'])
