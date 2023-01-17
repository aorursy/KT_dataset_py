import pandas as pd

import numpy as np



import re

import zipfile

import os
path = "/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/"
print(os.listdir(path)) # check the directories

print(len(os.listdir(path+'neg/'))) # check how many we have
# lets have a look at a file to check the review

example = []

with open(path+'neg/'+'112_1.txt') as f:

    example.append(f.read())

example
data = {}

for dir in ['pos', 'neg']: 

    data[dir] = []

    for file in os.listdir(path+dir+'/'):

        if file.endswith(".txt"):

            with open(path+dir+'/'+file) as f: # add encoding parameter?

                data[dir].append((f.read(), re.match(r'(?P<example>\d+)_(?P<rating>\d+)', file)['rating'])) # store a tuple of the text and the rating from the file name
postext, posrev = zip(*data['pos'])

negtext, negrev = zip(*data['neg'])



reviews = pd.concat([

    pd.DataFrame({"text":postext, "rating":posrev, "label":1}),

    pd.DataFrame({"text":negtext, "rating":negrev, "label":0})

], ignore_index=True) # now we have our df
reviews.info()
reviews = reviews.astype({'rating': 'category', 'label': 'category'})
reviews['text'][0] # this review (and many others) has html tags in the text. These could potentially be useful for a complex model so could regularise them but for simple models will just clean them
# also load the test set here

path = "/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/"

data = {}

for dir in ['pos', 'neg']: 

    data[dir] = []

    for file in os.listdir(path+dir+'/'):

        if file.endswith(".txt"):

            with open(path+dir+'/'+file) as f: # add encoding parameter?

                data[dir].append((f.read(), re.match(r'(?P<example>\d+)_(?P<rating>\d+)', file)['rating'])) # store a tuple of the text and the rating from the file name



postext, posrev = zip(*data['pos'])

negtext, negrev = zip(*data['neg'])



test_reviews = pd.concat([

    pd.DataFrame({"text":postext, "rating":posrev, "label":1}),

    pd.DataFrame({"text":negtext, "rating":negrev, "label":0})

], ignore_index=True) # now we have our df

    
%matplotlib inline

import matplotlib.pyplot as plt
reviews['label'].value_counts().plot(kind='bar') # confirm balanced dataset
reviews['rating'].value_counts().plot(kind='bar') # the data is a bit skewed towards the extreme values of 1 and 10
# word count hist???
reviews.drop_duplicates(keep='first',inplace=True)

test_reviews.drop_duplicates(keep='first',inplace=True)
reviews.dropna(inplace=True)

test_reviews.dropna(inplace=True)
from sklearn.base import BaseEstimator, TransformerMixin
class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values
from bs4 import BeautifulSoup
class HTMLCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, parser='lxml'):

        self.parser = parser

    def fit(self, y=None):

        return self

    def transform(self, X):

        return [BeautifulSoup(doc, self.parser).text for doc in X]
import spacy
class SpacyPipe(BaseEstimator, TransformerMixin):

    def __init__(self, batch_size, no_cores):

        self.batch_size = batch_size

        self.no_cores = no_cores

        self.nlp = spacy.load('en')

    def fit(self, y=None):

        return self

    def transform(self, X):

        if len(X) == 0:

            return [self.nlp(X),]

        return self.nlp.pipe(X)
class PreProcessor(BaseEstimator, TransformerMixin):

    def __init__(self, lemma:bool=True, lower:bool=True, stop:bool=True, rules:list=['PUNCT', 'SYM', 'X']):

        self.lemma = lemma

        self.lower = lower

        self.stop = stop

        for rule in rules:

            assert isinstance(rule, str), "Please pass a list of strings"

        self.rules = rules



    def lem(self, token):

        if self.lemma:

            return token.lemma_

        else:

            return token.text

    

    def low(self, text):

        if self.lower:

            return text.lower()

        else:

            return text

        

    def check_stop(self, token):

        if self.stop:

            return token.is_stop

        else:

            return True

    

    def fit(self, y=None):

        return self

    

    def transform(self, X):

        return [

                    ' '.join([

                    self.low(self.lem(n)) for n in doc 

                    if not 

                     (

                         self.check_stop(n) or 

                         any([(n.pos_ is rule) for rule in self.rules])

                     )

                    ])

                for doc in X

                ]
from sklearn.pipeline import Pipeline

import multiprocessing

cores = multiprocessing.cpu_count()
prepro_pipe = Pipeline([

    ('colselector', ColumnSelector(attribute_names='text')),

    ('html_cleaner', HTMLCleaner(parser='lxml')),

    ('spacy_generator', SpacyPipe(batch_size=50, no_cores=cores)),

    ('preprocessor', PreProcessor(lemma=True, lower=True, stop=True, rules=['PUNCT', 'SYM', 'X']))

])
prepro_reviews = prepro_pipe.fit_transform(reviews)
reviews['pre_processed'] = prepro_reviews
reviews.head()
from sklearn.feature_extraction.text import CountVectorizer
bagofwords_vec = CountVectorizer(ngram_range=(1,3), min_df=2) # note we will fit this to training data, for test data only transform
bow = bagofwords_vec.fit_transform(reviews['pre_processed'])
reviews['feat_bow'] = list(bow) # list as we want to convert to rows
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(ngram_range=(1,3), min_df=2)
tfidf = tfidf_vec.fit_transform(reviews['pre_processed'])
reviews['feat_tfidf'] = list(tfidf)
!pip install wget
#os.remove('/kaggle/working/word2vec')
#import wget

#url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

#wget.download(url, '/kaggle/working')
#import gzip

#with gzip.open("/kaggle/working/GoogleNews-vectors-negative300.bin.gz", 'rb') as f_in:

#    wtovmodel = f_in.read()

# now model is downloaded you can pass it into gensim to use the model and convert your df
from gensim import models

#wtovmodel = models.KeyedVectors.load_word2vec_format(wtovmodel, binary=True)
from gensim.models import FastText