import numpy as np

import pandas as pd

import random

import string

from bs4 import BeautifulSoup

from urllib import request

import re

import nltk
html = request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing').read()

article_html = BeautifulSoup(html,'lxml')

article_paragraphs = article_html.find_all('p')

article_text = ''

for para in article_paragraphs:

    article_text += para.text
corpus = nltk.sent_tokenize(article_text)
for i in range(len(corpus)):

    corpus[i] = corpus[i].lower()

    corpus[i] = re.sub('\W',' ',corpus[i])

    corpus[i] = re.sub('\s+',' ',corpus[i])
print(len(corpus))

s_corpus = ' '.join(corpus)

word_freq = nltk.FreqDist(nltk.word_tokenize(s_corpus)).most_common(200)
most_freq = dict(word_freq)
sentence_vectors = []

for sent in corpus:

    sent_tokens = nltk.word_tokenize(sent)

    sent_vec = []

    for val in most_freq:

        if val in sent_tokens:

            sent_vec.append(1)

        else:

            sent_vec.append(0)

    sentence_vectors.append(sent_vec)
sentence_vectors = []

for sent in corpus:

    sent_words = nltk.word_tokenize(sent)

    sent_vec = []

    for val in most_freq:

        if val in sent_words:

            sent_vec.append(1)

        else:

            sent_vec.append(0)

    sentence_vectors.append(sent_vec)

            
np.asarray(sentence_vectors)