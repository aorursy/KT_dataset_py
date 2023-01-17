import re

import nltk

import pandas as pd

import numpy as np

from nltk.stem.snowball import SnowballStemmer

import matplotlib.pyplot as plt

from string import digits

from gensim.models import Word2Vec

import os

import gensim

from snowballstemmer import TurkishStemmer

from keras.preprocessing.sequence import pad_sequences
turkishStemmer = TurkishStemmer()



WPT = nltk.WordPunctTokenizer()

stop_word_list = nltk.corpus.stopwords.words('turkish')
model = gensim.models.KeyedVectors.load_word2vec_format('../input/fasttext-aligned-word-vectors/wiki.tr.align.vec', limit=200000)
sentence = []

label = []



with open("../input/cyberbullyingturkishdataset/erkek_hakaret_var.txt", encoding="utf8") as f:

    for line in f:

        line = re.sub('[.\n]', '', line)

        sentence.append(line)

        label.append('Yes')



with open("../input/cyberbullyingturkishdataset/erkek_hakaret_yok.txt", encoding="utf8") as f:

    for line in f:

        line = re.sub('[.\n]', '', line)

        sentence.append(line)

        label.append('No')



with open("../input/cyberbullyingturkishdataset/kadn_hakaret_yok.txt", encoding="utf8") as f:

    for line in f:

        line = re.sub('[.\n]', '', line)

        sentence.append(line)

        label.append('No')



with open("../input/cyberbullyingturkishdataset/kadn_hakaret_var.txt", encoding="utf8") as f:

    for line in f:

        line = re.sub('[.\n]', '', line)

        sentence.append(line)

        label.append('Yes')
data = pd.DataFrame(list(zip(sentence, label)), columns = ['sentence', 'is_bullying'])

data = data.sample(frac=1).reset_index(drop=True)

y = data['is_bullying'].values



display(data)
processed = []

count = 0

value = 1

data_dict = {}

sequences = []



for text in data['sentence']:

    text = text.lower()



    tokens = WPT.tokenize(text)

    processed.append([])

    sequences.append([])

    for word in tokens:



        word = turkishStemmer.stemWord(word)

        if word in model.vocab:

            if word in data_dict:

                processed[count].append(word)

                sequences[count].append(data_dict.get(word))

            else:

                data_dict.update({word: value})

                processed[count].append(word)

                sequences[count].append(value)

                value += 1



        else:

            print("not in model")

    count += 1
list_len = [len(i) for i in sequences]

print(max(list_len))



data2 = pad_sequences(sequences, padding = 'post', maxlen = max(list_len))

print('Shape of data tensor:', data2.shape)
VALIDATION_SPLIT = 0.2



num_validation_samples = int(VALIDATION_SPLIT*data.shape[0])

x_train = data[: -num_validation_samples]

y_train = y[: -num_validation_samples]

x_val = data[-num_validation_samples: ]

y_val = y[-num_validation_samples: ]