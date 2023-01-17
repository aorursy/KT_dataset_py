# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import nltk

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import TruncatedSVD
data = pd.read_csv("../input/vr_titles.csv", encoding = "utf8")

titles = data['Software Title'].values 
from nltk.corpus import stopwords as sw

stopwords = sw.words("english") 

wordnet_lemmatizer = WordNetLemmatizer()

def vr_tokenizer(s):

    s = s.lower() # downcase

    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)

    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful

    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form

    tokens = [t for t in tokens if t not in stopwords] # remove stopwords

    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"

    return tokens
word_index_map = {}

current_index = 0

all_tokens = []

all_titles = []

index_word_map = []

for title in titles:

        title = title.encode('ascii', 'ignore') # this will throw exception if bad characters

        all_titles.append(title)

        tokens = vr_tokenizer(title.decode('utf-8'))

        all_tokens.append(tokens)

        for token in tokens:

            if token not in word_index_map:

                word_index_map[token] = current_index

                current_index += 1

                index_word_map.append(token)

def tokens_to_vector(tokens):

    x = np.zeros(len(word_index_map))

    for t in tokens:

        i = word_index_map[t]

        x[i] = 1

    return x
N = len(all_tokens)

D = len(word_index_map)

X = np.zeros((D, N)) # terms will go along rows, documents along columns

i = 0

for tokens in all_tokens:

    X[:,i] = tokens_to_vector(tokens)

    i += 1

    

print (X)
svd = TruncatedSVD()

Z = svd.fit_transform(X)

plt.scatter(Z[:,0], Z[:,1])

for i in range(D):

    plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))

plt.show()