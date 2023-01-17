import numpy as np

import pandas as pd

import json

import os



titles = []

abstracts = []

texts = []

for dirname, _, filenames in os.walk("/kaggle/input/CORD-19-research-challenge/"):

    for filename in filenames:

        file = os.path.join(dirname, filename)

        if file.split(".")[-1] == "json":

            with open(file,"r")as f:

                doc = json.load(f)

                titles.append(doc["metadata"]["title"])

                abstracts.append(" ".join([item["text"] for item in doc["abstract"]]))

                texts.append(" ".join([item["text"] for item in doc["body_text"]]))
from nltk.tokenize import sent_tokenize

from gensim.utils import simple_preprocess

from gensim.sklearn_api.phrases import PhrasesTransformer

from gensim.models import Word2Vec



sentences = []

for text in texts:

    sentences += [simple_preprocess(sentence) for sentence in sent_tokenize(text)]

bigrams = PhrasesTransformer(min_count=20, threshold=100)

model = Word2Vec(bigrams.fit_transform(sentences), 

                 size=100, 

                 window=10, 

                 min_count=10,

                 sg=1,

                 negative=5,

                 max_final_vocab=30000, 

                 workers=4, 

                 iter=4)
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



main_keywords = ["transmission", "incubation", "genetics"]

vectors = []

keywords = []

for k in main_keywords:

    vectors.append(model.wv[k])

    keywords.append(k)

    for w, s in model.wv.most_similar(positive=k, topn=5):

        vectors.append(model.wv[w])

        keywords.append(w)

svd = TruncatedSVD(n_components=2)

X = svd.fit_transform(vectors)

plt.figure(figsize=(15, 10))

colors = ["red", "blue", "green"]

for i in range(3):

    plt.scatter(X[i*6:(i+1)*6,0], X[i*6:(i+1)*6,1], c=colors[i])

i = 0

for keyword in keywords:

    plt.annotate(keywords[i], (X[i,0], X[i,1]))

    i += 1 