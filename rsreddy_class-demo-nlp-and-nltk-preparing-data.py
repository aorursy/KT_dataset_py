# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import nltk
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
text="Mary had a little lamb. Her fleece was white as snow"
# Functions to break text into sentences/words
from nltk.tokenize import word_tokenize, sent_tokenize
sents=sent_tokenize(text)
# Text tokenized into a list of sentences
print(sents)

# One thing to note here: even punctuation marks are considered as 'word'
words=[word_tokenize(sent) for sent in sents]
print(words)
# Remove Stopwords
from nltk.corpus import stopwords
from string import punctuation
customStopWords=set(stopwords.words('english')+list(punctuation))

wordsWOStopwords=[word for word in word_tokenize(text) if word not in customStopWords]
print(wordsWOStopwords)
# N-Grams - where words are together. Most important bigrams will be on top
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWOStopwords)
sorted(finder.ngram_fd.items())

# Stemming
text2 = "Mary closed on closing night when she was in the mood to close."
from nltk.stem.lancaster import LancasterStemmer
st=LancasterStemmer()
stemmedWords=[st.stem(word) for word in word_tokenize(text2)]
print(stemmedWords)

# Tagging
nltk.pos_tag(word_tokenize(text2))

# Disambiguating Word Meaning
from nltk.corpus import wordnet as wn
for ss in wn.synsets('bass'):
    print(ss, ss.definition())
from nltk.wsd import lesk
sense1 = lesk(word_tokenize("Sing in a lower tone, along with the bass"),'bass')
print(sense1, sense1.definition())    
sense2 = lesk(word_tokenize("This sea bass was really hard to catch"),'bass')
print(sense2, sense2.definition())
