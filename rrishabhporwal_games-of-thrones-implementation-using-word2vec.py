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
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# nltk.download("punkt") #Pretrained Tokenizer

# nltk.download("stopwords") #Words like a, the, an, of etc


book_filenames = sorted(glob.glob("../input/*.txt"))

# book_filenames = sorted(glob.glob("data/*.txt"))
print("Books Found :")

book_filenames
corpus_raw = u""

for book_filename in book_filenames:

    print("Reading '{0}'...".format(book_filename))

    with codecs.open(book_filename, "r", "UTF-8") as book_file:

        corpus_raw += book_file.read()

    

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

vector = model.wv['khaleesi']
vector
# Most similar words

model.wv.most_similar("khaleesi")
model.wv.most_similar("aerys")
model.wv.most_similar("direwolf")
#distance, similarity, and ranking

def nearest_similarity_cosmul(start1, end1, end2):

    similarities = model.wv.most_similar_cosmul(

        positive=[end2, start1],

        negative=[end1]

    )

    start2 = similarities[0][0]

    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))

    return start2
nearest_similarity_cosmul("stark", "winterfell", "riverrun")

nearest_similarity_cosmul("jaime", "sword", "wine")

nearest_similarity_cosmul("arya", "nymeria", "dragons")
X = model[model.wv.vocab]
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)

plt.figure(figsize=(12,5))

for i, word in enumerate(words):

    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()