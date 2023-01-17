# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

import keras

from keras.models import Sequential

import nltk

import re

from gensim.models import Word2Vec

from gensim.models.word2vec import LineSentence

from collections import defaultdict





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
with open("../input/Tweets.csv", "r") as f:

    df = pd.read_csv(f)

df = df.drop(['tweet_id', 'negativereason', 'airline', 'airline_sentiment_gold', 'name',

             'negativereason_gold', 'negativereason_confidence', 'retweet_count', 'tweet_coord',

             'tweet_created', 'tweet_location', 'user_timezone'], axis=1)
df = df[df['airline_sentiment_confidence'] > 0.75]

X, y = df.iloc[:, 1:], df.iloc[:, :1]

print("number of samples: {}, number of targets: {}".format(len(X), len(y)))

le = LabelEncoder()

y = le.fit_transform(np.array(y).ravel())

print(X[:10])

print(y)
print(X['text'][:140:7])



def text_cleaner(text):

    text = re.sub(r'@\w+', '_TN', text)

    text = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', '_EM', text)

    text = re.sub(r'\w+:\/\/\S+', r'_U', text)

    text = text.replace('"', ' ')

    text = text.replace('\'', ' ')

    text = text.replace('_', ' ')

    text = text.replace('-', ' ')

    text = text.replace('\n', ' ')

    text = text.replace('\\n', ' ')

    text = text.replace('\'', ' ')

    text = re.sub(' +', ' ', text)

    text = text.replace('\'', ' ')

    text = re.sub(r'([^!\?])(\?{2,})(\Z|[^!\?])', r'\1 _BQ\n\3', text)

    text = re.sub(r'([^\.])(\.{2,})', r'\1 _SS\n', text)

    text = re.sub(r'([^!\?])(\?|!{2,})(\Z|[^!\?])', r'\1 _BX\n\3', text)

    text = re.sub(r'([^!\?])\?(\Z|[^!\?])', r'\1 _Q\n\2', text)

    text = re.sub(r'([^!\?])!(\Z|[^!\?])', r'\1 _X\n\2', text)

    text = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL', text)

    text = re.sub(r'(\w+)\.(\w+)', r'\1\2', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'([#%&\*\$]{2,})(\w*)', r'\1\2 _SW', text)

    text = re.sub(r' [8x;:=]-?(?:\)|\}|\]|>){2,}', r' _BS', text)

    text = re.sub(r' (?:[;:=]-?[\)\}\]d>])|(?:<3)', r' _S', text)

    text = re.sub(r' [x:=]-?(?:\(|\[|\||\\|/|\{|<){2,}', r' _BF', text)

    text = re.sub(r' [x:=]-?[\(\[\|\\/\{<]', r' _F', text)

    text = re.sub(r'\s\s', r' ', text)

    return text



X['text'] = X['text'].apply(text_cleaner)



print(X['text'][:140:7])
X['text'] = X['text'].apply(lambda x: x.split())

print(X['text'])
X['text'].apply(lambda x: LineSentence(x))

print(X['text'])
# this part not currently used



def chunker(sentence):

    tokens = nltk.word_tokenize(' '.join(sentence))

    tagged = nltk.pos_tag(tokens)

    entities = nltk.chunk.ne_chunk(tagged)

    return entities



def get_content_words(chunks):

    content_words = ['NN', 'NNS', 'JJ']

    exclude = ['i', 'you', 'he', 'she', 'it', 'them', 'us', 'retweet', 'rt']

    prefixes = ('@', '\\', '?', 'http', '/', '#', 'rt')

    content = [x for x in chunks if type(x[-1]) == str and x[-1] in content_words]

    content = [x[0] for x in content if x[0].lower() not in exclude]

    content = [x for x in content if not x.lower().startswith(prefixes)]

    return content
sentences = X['text'].values

model = Word2Vec(sentences, min_count=10)

print("vocabulary size: {} words".format(len(model.vocab)))
print(model.similar_by_word('disappointed'))
# see http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

class Word_Vectoriser(object):

    

    def __init__(self, w2v):

        self.w2v = w2v

        self.dim = len(list(w2v.values())[0])

        self.word2weight = None

    

    def fit(self, X, y):

        tfidf = TfidfVectorizer(analyzer=lambda x: x)

        tfidf.fit(X)

        max_idf = max(tfidf.idf_)

        self.word2weight = defaultdict(

            lambda: max_idf,

            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    

    def get_params(self, *args, **kwargs):

        return self

    

    def transform(self, X):

        return np.array([

            np.mean([self.w2v[w] * self.word2weight[w] for w in words if w in self.w2v]

                 or [np.zeros(self.dim)], axis=0)

            for words in X])



X_train, X_test, y_train, y_test = train_test_split(X['text'], y, test_size=0.20, random_state=1)

w2v = dict(zip(model.index2word, model.syn0))
print(X_train.shape)

print(y_train.shape)
clf = SVC()

parameters = {'C': [10, 100, 1000, 10000], 'gamma': [0.01, 0.001, 0.0001]}

grid_search = GridSearchCV(clf, parameters, n_jobs=8, verbose=1)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_, grid_search.best_score_)
pipeline = Pipeline([('w2v', Word_Vectoriser(w2v)),

                     ('svm', SVC(C=10000, gamma=0.0001))])



pipeline.fit(X_train, y_train)
# make pipeline for vectorisation and classification steps

pipeline = Pipeline([('w2v', Word_Vectoriser(w2v)),

                     ('svm', SVC(C=10000, gamma=0.01))])
pipeline.score(X_test, y_test)
print(le.inverse_transform([0, 1, 2]))

# test positives

indices = np.argwhere(y_test == 2)

X_pos = [X_test[i] for i in indices]

print(X_pos)