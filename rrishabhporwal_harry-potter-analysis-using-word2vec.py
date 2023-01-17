#Goal : create word vector from Game of Throne dataset

from __future__ import absolute_import, division, print_function # for dependency python 2 to 3

# For word encoding

import codecs

# Regex

import glob

# Concurrency

import multiprocessing

# Dealing with operating system like reading files

import os

# Pretty Printing

import pprint

# Regular Expression

import re

# Natural Language  Toolkit

import nltk

from nltk.corpus import stopwords

# WOrd 2 vec

from gensim.models import Word2Vec

# Dimensional Reductionality

import sklearn.manifold

#math

import numpy as np

#plotting

import matplotlib.pyplot as plt

#data processing 

import pandas as pd

# Data Visualization

import seaborn as sns



%matplotlib inline
book_filenames = sorted(glob.glob("../input/*.txt"))
print("Books Found :")

book_filenames
corpus_raw = ""

for book_filename in book_filenames:

    print("Reading '{0}'...".format(book_filename))

    with open(book_filename, "rb") as infile:

        corpus_raw += str(infile.read())

        

        print("Corpus is now {0} characters long". format(len(corpus_raw)))

        print()

        
text = corpus_raw



# Preprocessing the data

text = re.sub(r'\[[0-9]*\]',' ',text)

text = re.sub(r'\s+',' ',text)

text = text.lower()

text = text.strip()

text = re.sub(r'\d',' ',text)

text = re.sub(r'\s+',' ',text)
# Preparing the dataset

sentences = nltk.sent_tokenize(text)
sentences
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
sentences
for i in range(len(sentences)):

    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
sentences
# Training the Word2Vec model

model = Word2Vec(sentences, min_count=1)
model
words = model.wv.vocab
words
# Finding Word Vectors

vector = model.wv['harry']
vector
# Most similar words

similar = model.wv.most_similar('mcgonagall')
similar
#distance, similarity, and ranking

def nearest_similarity_cosmul(start1, end1, end2):

    similarities = model.wv.most_similar_cosmul(

        positive=[end2, start1],

        negative=[end1]

    )

    start2 = similarities[0][0] 

    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))

    return start2
nearest_similarity_cosmul("harry", "professor", "snape")

nearest_similarity_cosmul("dumbledore", "elder", "wand")

nearest_similarity_cosmul("lupin", "james", "sirius")
X = model[model.wv.vocab]
X
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)

plt.figure(figsize=(12,5))

for i, word in enumerate(words):

    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()