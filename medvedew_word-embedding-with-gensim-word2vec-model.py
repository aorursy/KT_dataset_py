import pandas as pd

import numpy as np

import re

from gensim.models import Word2Vec

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.decomposition import TruncatedSVD

from sklearn.manifold import TSNE

import random

from collections import Counter
TOKEN_RE = re.compile(r'[\w]+')

def tokenize_text_simple_regex(txt, min_token_size=3):

    txt = str(txt).lower()

    all_tokens = TOKEN_RE.findall(txt)

    return [token for token in all_tokens if len(token) >= min_token_size]
def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):

    return [tokenizer(text, **tokenizer_kwargs) for text in texts]
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

corpus = tokenize_corpus(list(train['text']))
random.seed(2020)

random.shuffle(data)
model = Word2Vec(

        sentences=corpus,

        size=50, 

        window=5, 

        min_count=2, 

        sg=1, #skip-gram

        negative=10, 

        iter=500, 

        seed=2020,

        workers=6

        )
def plot_vectors(vectors, labels, how='tsne', ax=None):

    if how == 'tsne':

        projections = TSNE().fit_transform(vectors)

    elif how == 'svd':

        projections = TruncatedSVD().fit_transform(vectors)



    x = projections[:, 0]

    y = projections[:, 1]



    ax.scatter(x, y)

    for cur_x, cur_y, cur_label in zip(x, y, labels):

        ax.annotate(cur_label, (cur_x, cur_y))
test_words = ['http','https','fire','video','disaster','burning','youtube']

gensim_words = [w for w in test_words if w in model.wv.vocab]

gensim_vectors = np.stack([model.wv[w] for w in gensim_words])
fig, ax = plt.subplots()

fig.set_size_inches((10, 10))

plot_vectors(gensim_vectors, test_words, how='svd', ax=ax)
model.save("w2v.model")