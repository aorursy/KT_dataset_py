# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#from yellowbrick.text import TSNEVisualizer

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer #for term frequency

from nltk import word_tokenize          

from nltk.stem import WordNetLemmatizer 

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.naive_bayes import MultinomialNB



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

class LemmaTokenizer(object):

    def __init__(self):

        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):

        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
corpus = ['Python is a versatile programming language','R and Python are open source','Python Rocks','Python and R have text-processing capability','Prorgramming in Python is easy','Python and R have rich libraries']

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

wnames= vectorizer.get_feature_names()

dtm=X.toarray()

np.asarray(dtm)
def getWordNames(corpus,ind):

    #  Takes the corpus as the input

    #  ind allows to specify whether to use term frequency or term frequency inverse document frequency

    if ind=='C':

        vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())

    else:

        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),min_df=0.01, norm='l2', stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=False)

    X = vectorizer.fit_transform(corpus)

    wnames= vectorizer.get_feature_names()

    return wnames



def getTermDocumentMatrix(corpus,ind):

    #  Takes the corpus as the input

    #  ind allows to specify whether to use term frequency or term frequency inverse document frequency

    if ind=='C':

        vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())

    else:

        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),min_df=0.01, norm='l2', stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=False)

    X = vectorizer.fit_transform(corpus)

    dtm=X.toarray()

    np.asarray(dtm)

    dtm=np.transpose(dtm)

    return dtm
def distmatrix (dtm):

    # Takes the document term matrix as input and returns the distance matrix

    n,p = dtm.shape

    dist = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            x, y = dtm[i, :], dtm[j, :]

            dist[i, j] = np.sqrt(np.sum((x - y)**2))

    return dist
# Finding nearest word

def closest_word(w1,wnames,dist):

    # Takes 3 parameters as input, word, name of all words or vocabulary, 

    ind= wnames.index(w1)

    distv=dist[ind,:]

    valid_idx = np.where(distv > 0)[0]

    out = valid_idx[distv[valid_idx].argmin()]

    print('nearest word is ' + wnames[out])
# Creating Corpus as a python list

corpus = ['Python is a versatile programming language','R and Python are open source','Python Rocks','Python and R have text-processing capability','Prorgramming in Python is easy','Python and R have rich libraries']

wnames=getWordNames(corpus,'T')

print(wnames)

dtm=getTermDocumentMatrix(corpus,'T')

print(dtm)

dist=distmatrix(dtm)

closest_word('python',wnames,dist)
nRowsRead =1000

filepath="/kaggle/input/brown-corpus/brown.csv"

df = pd.read_csv(filepath,encoding='iso-8859-1',nrows=nRowsRead)

df.head()
corpus = df.tokenized_text

wnames=getWordNames(corpus,'T')

print(wnames)

dtm=getTermDocumentMatrix(corpus,'T')

dist=distmatrix(dtm)

closest_word('wife',wnames,dist)
corpus = ['Chinese Beijing Chinese','Chinese Chinese Shanghai','Chinese Macao','Tokyo Japan Chinese','Chinese Chinese Chinese Tokyo Japan']

labels=['c','c','c','j']

wnames=getWordNames(corpus,'C')

print(wnames)

dtm=getTermDocumentMatrix(corpus,'C')

# Transposing to get the documents in rows

dtm=np.transpose(dtm)

trn=dtm[0:4,:]

clf = MultinomialNB()

clf.fit(trn, labels)

# To make a two dimensional array out of the test set, will not be required if test set has multiple rows

tst=dtm[4,:].reshape(1, -1)

clf.predict(tst)
from nltk.corpus import reuters

from nltk import bigrams, trigrams

from collections import Counter, defaultdict

model = defaultdict(lambda: defaultdict(lambda: 0))

 

for sentence in reuters.sents():

    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):

        model[(w1, w2)][w3] += 1



for w1_w2 in model:

    total_count = float(sum(model[w1_w2].values()))

    for w3 in model[w1_w2]:

        model[w1_w2][w3] /= total_count

 

print(model["what", "the"]["economists"])

print(model["what", "the"]["nonexistingword"])

print(model[None, None]["The"])
import gensim

from gensim.models import Word2Vec

from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

from gensim.models import KeyedVectors

# Load vectors directly from the file

model = KeyedVectors.load_word2vec_format('/kaggle/input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin', binary=True)

# Access vectors for specific words with a keyed lookup:

vector = model['easy']

# see the shape of the vector (300,)

vector.shape
model.similarity('easy','simple')

model.similarity('joy','sorry')

model.most_similar('simple')

model.get_vector('simple')