#Load required libraries

import numpy as np

import pandas as pd

#For displaying complete rows info

pd.options.display.max_colwidth=500

import tensorflow as tf

import spacy

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from collections import Counter

from functools import partial

from tqdm import tqdm_notebook

print(tf.__version__)

#Load data into pandas dataframe

df=pd.read_csv("../input/articles.csv",encoding="utf8")
df.head(2)
print(df["title"][0],"\n",df["text"][0])
#Properly formatted data removing nans

df.drop_duplicates(subset=["text"],inplace=True)

df.dropna(inplace=True)

df.reset_index(drop=True,inplace=True)
import gensim

import string

import re

from itertools import chain
# Portuguese word and sent tokenizers

word_tokenize_pt = partial(word_tokenize, language='portuguese')

# sent_tokenize_pt = partial(sent_tokenize, language='portuguese')

filter_small_words = lambda words: list(filter(lambda w: len(w) > 2, words))
articles_tokens = map(word_tokenize_pt, tqdm_notebook(df['text'].str.lower()))

# articles_tokens = map(word_tokenize_pt, articles_tokens)

articles_tokens = map(filter_small_words, articles_tokens)

articles_tokens = list(articles_tokens)
model = gensim.models.Word2Vec(articles_tokens, min_count=5, size=100, workers=4, window=5)
model.wv.most_similar("lula")
from collections import Counter



cnt = Counter({k:v.count for k, v in model.wv.vocab.items()})
all_sents = chain(*map(sent_tokenize, df['text']))
all_sents = list(all_sents)

len(all_sents)
def sent2vector(sent):

    words = word_tokenize(sent.lower())

    

    # Here we weight-average each word in sentence by 1/log(count[word])

    emb = [model[w] for w in words if w in model]

    weights = [1./cnt[w] for w in words if w in model]

    

    if len(emb) == 0:

        return np.zeros(100, dtype=np.float32)

    else:

        return np.dot(weights, emb) / np.sum(weights)
sent_vectors = np.array(list(map(sent2vector, tqdm_notebook(all_sents))))
sent_vectors.shape
from sklearn.neighbors import KDTree

kdtree = KDTree(sent_vectors)
def search(sent, k=3):

    sent_vec = sent2vector(sent)

    closest_sent = kdtree.query(sent_vec[None], k)[1][0]

    

    return [all_sents[i] for i in closest_sent]
search('desemprego no brasil', 3)
search('futebol barcelona', 3)
search('operação lava jato', 3)
search('avanços na medicina do câncer', 3)