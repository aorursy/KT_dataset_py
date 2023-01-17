from nltk.corpus import nps_chat

import nltk

import numpy as np

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import string

from nltk.corpus import wordnet

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn import neighbors

from sklearn.model_selection import RandomizedSearchCV

import scipy

from sklearn.metrics import *

import pickle

from nltk.corpus import sentence_polarity



def my_tokenizer(doc):

    words = word_tokenize(doc)

    pos_tags = pos_tag(words)

    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]

    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]

    lemmas = []

    for w in non_punctuation:

        if w[1].startswith('J'):

            pos = wordnet.ADJ

        elif w[1].startswith('V'):

            pos = wordnet.VERB

        elif w[1].startswith('N'):

            pos = wordnet.NOUN

        elif w[1].startswith('R'):

            pos = wordnet.ADV

        else:

            pos = wordnet.NOUN

        lemmas.append(lemmatizer.lemmatize(w[0], pos))

    return lemmas



class SVDDimSelect(object):

    def fit(self, X, y=None):        

        try:

            self.svd_transformer = TruncatedSVD(n_components=round(X.shape[1]/2))

            self.svd_transformer.fit(X)

            cummulative_variance = 0.0

            k = 0

            for var in sorted(self.svd_transformer.explained_variance_ratio_)[::-1]:

                cummulative_variance += var

                if cummulative_variance >= 0.5:

                    break

                else:

                    k += 1

            self.svd_transformer = TruncatedSVD(n_components=k)

        except Exception as ex:

            print(ex)

        return self.svd_transformer.fit(X)

    def transform(self, X, Y=None):

        return self.svd_transformer.transform(X)

    def get_params(self, deep=True):

        return {}



x_data_neg = []

x_data_neg.extend([' '.join(post) for post in sentence_polarity.sents(categories=['neg'])])

y_data_neg = [0] * len(x_data_neg)

x_data_pos = []

x_data_pos.extend([' '.join(sent) for sent in sentence_polarity.sents(categories=['pos'])])

y_data_pos = [1] * len(x_data_pos)

x_data_full = x_data_neg[:500] + x_data_pos[:500]

y_data_full = y_data_neg[:500] + y_data_pos[:500]

x_data = np.array(x_data_full, dtype=object)

y_data = np.array(y_data_full)

train_indexes = np.random.rand(len(x_data)) < 0.80

x_data_train = x_data[train_indexes]

y_data_train = y_data[train_indexes]

x_data_test = x_data[~train_indexes]

y_data_test = y_data[~train_indexes]

stopwords_list = stopwords.words('english')

lemmatizer = WordNetLemmatizer()
# CRIADO PELO PROFESSOR

!pip install glove_python

from gensim.models import Word2Vec

#sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.

class Word2VecTransformer(object):

    

    ALGO_SKIP_GRAM=1

    ALGO_CBOW=2    

    

    def __init__(self, algo=1):

        self.algo = algo

    

    def fit(self, X, y=None):     

        X = [nltk.word_tokenize(x) for x in X]

        self.word2vec = Word2Vec(X, min_count=2, sg=self.algo)

        # Pegamos a dimensão da primeira palavra, para saber quantas dimensões estamos trabalhando,

        # assim podemos ajustar nos casos em que aparecerem palavras que não existirem no vocabulário.

        first_word = next(iter(self.word2vec.wv.vocab.keys()))

        self.num_dim = len(self.word2vec[first_word])       

        return self

    

    def transform(self, X, Y=None):        

        X = [nltk.word_tokenize(x) for x in X]

        

        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.num_dim)], axis=0) 

                         for words in X])

    def get_params(self, deep=True):

        return {}
#CRIADO EM SALA



from glove import Corpus, Glove



class glovetransformer(object):

    ALGO_SKIP_GRAM=1

    ALGO_CBOW=2    

    def __init__(self, algo=1):

        self.algo = algo

        

    def fit(self, X, y=None):     

        X = [nltk.word_tokenize(x) for x in X]

        self.corpus = Corpus() 

        self.corpus.fit(X, window=10)

        self.num_dim = 50

        self.glove = Glove(no_components=50, learning_rate=0.05) #5 mt pouco pois 50

        self.glove.fit(self.corpus.matrix, epochs=30, no_threads=1, verbose=True)

        self.glove.add_dictionary(self.corpus.dictionary)

        return self



    def transform(self, X, Y=None):        

        X = [nltk.word_tokenize(x) for x in X]

        return np.array([np.mean([self.glove.word_vectors[self.glove.dictionary[w]] for w in words if w in self.corpus.dictionary.keys()] or [np.zeros(self.num_dim)], axis=0) for words in X])

     

    def get_params(self, deep=True):

        return {}
# Pipeline para os 3  



clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



skgram_pipeline = Pipeline([('skg', Word2VecTransformer(algo=1)), ('clf', clf)]) 

cbow_pipeline = Pipeline([('cbo', Word2VecTransformer(algo=2)), ('clf', clf)]) 

glove_pipeline = Pipeline([('glo', glovetransformer()), ('clf', clf)]) 



par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}



skgram_hyperpar_selector = RandomizedSearchCV(skgram_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20) #njobs > 1 no kaggle nao permite pois nao consegue rodar mais de 1 thread

cbow_hyperpar_selector = RandomizedSearchCV(cbow_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20) #njobs > 1 no kaggle nao permite pois nao consegue rodar mais de 1 thread

glove_hyperpar_selector = RandomizedSearchCV(glove_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20) #njobs > 1 no kaggle nao permite pois nao consegue rodar mais de 1 thread





skgram_hyperpar_selector.fit(X=x_data_train, y=y_data_train)

cbow_hyperpar_selector.fit(X=x_data_train, y=y_data_train)

glove_hyperpar_selector.fit(X=x_data_train, y=y_data_train)
from sklearn.metrics import classification_report

skgram_y_pred = skgram_hyperpar_selector.predict(x_data_test)

print(classification_report(y_data_test, skgram_y_pred, target_names=['neg','pos']))

cbow_y_pred = cbow_hyperpar_selector.predict(x_data_test)

print(classification_report(y_data_test, cbow_y_pred, target_names=['neg','pos']))

glove_y_pred = glove_hyperpar_selector.predict(x_data_test)

print(classification_report(y_data_test, glove_y_pred, target_names=['neg','pos']))