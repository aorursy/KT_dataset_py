

!pip install spacy 

!python -m spacy download en



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import spacy 

nlp = spacy.load("en")
import spacy

from collections import Counter

nlp = spacy.load("en")

text = """Most of the outlay will be at home. No surprise there, either. While Samsung has expanded overseas, South Korea is still host to most of its factories and research engineers. """

doc = nlp(text)

words = [token.text for token in doc]

print (words)
import spacy

nlp = spacy.load("en")

text = """Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It was developed by Steven Bird and Edward Loper in the Department of Computer and Information Science at the University of Pennsylvania."""

text = nlp(text)

list(text.sents)
import spacy

from collections import Counter

nlp = spacy.load("en")

text = """Most of the outlay will be at home. No surprise there, either. While Samsung has expanded overseas, South Korea is still host to most of its factories and research engineers. """

doc = nlp(text)

#remove stopwords and punctuations

words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]

print (words)
import spacy

nlp = spacy.load('en')

text = """While Samsung has expanded overseas, South Korea is still host to most of its factories and research engineers. """

doc = nlp(text)

for token in doc:

    print(token, token.lemma_)

import spacy

from collections import Counter

nlp = spacy.load("en")

text = """Most of the outlay will be at home. No surprise there, either. While Samsung has expanded overseas, South Korea is still host to most of its factories and research engineers. """

doc = nlp(text)

#remove stopwords and punctuations

words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]

word_freq = Counter(words)

common_words = word_freq.most_common(5)

print (common_words)

import spacy

nlp = spacy.load("en")

text = """Natural Language Toolkit, or more commonly NLTK."""

text = nlp(text)

for w in text:

    print (w, w.pos_)
import spacy

nlp = spacy.load("en")

text = """Most of the outlay will be at home. No surprise there, either. While Samsung has expanded overseas, South Korea is still host to most of its factories and research engineers. """

text = nlp(text)

labels = set([w.label_ for w in text.ents]) 

for label in labels: 

    entities = [e.string for e in text.ents if label==e.label_] 

    entities = list(set(entities)) 

    print( label,entities)