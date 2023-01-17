

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



import nltk
from nltk.book import *
sents() # shows one sentence from each corpora
text7
sent7 # parsed out already
len(sent7)
len(text7) # how many words in total
len(set(text7)) # to see the number of unique words
list(set(text7))[:10] # to see first 10 unique words
# frequency of words

dist = FreqDist(text7) # FreqDist is a part of nltk library

len(dist) # returns the same number of unique words
vocab1 = list(dist.keys()) # to get the actual words

vocab1[:10]
dist['Vinken']
freqwords = [w for w in vocab1 if len(w)>5 and dist[w]>100]

freqwords
input1 = 'List listed lists listing listings'
words1 = input1.lower().split(" ") # don't want to distinguish "List" with "list"

words1 # normalization is done
# stemming is coming

porter = nltk.PorterStemmer() # to find to root form of any given word
[porter.stem(t) for t in words1]
udhr = nltk.corpus.udhr.words('English-Latin1')

udhr[:20]
[porter.stem(t) for t in udhr[:20]]

# resulting list: some words are not valid words
# lemmatization does stemming but all resulting words are valid

WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in udhr[:20]] # all words are valid
text11 = "Children shouldn't drink a sugary drink before bed."
text11.split(" ") # it's keeping the full stop with the word
nltk.word_tokenize(text11)
#nltk has a sentense splitter also

text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"

sentences = nltk.sent_tokenize(text12)

sentences
nltk.help.upenn_tagset('MD')
#apply nltk's word tokenizer

text13 = nltk.word_tokenize(text11)

text13
nltk.pos_tag(text13)
text14 = "Visiting aunts can be a nuisance"

text15 = nltk.word_tokenize(text14)

nltk.pos_tag(text15)
text15 = nltk.word_tokenize("Alice loves Bob")

grammar = nltk.CFG.fromstring("""

S -> NP VP

VP -> V NP

NP -> 'Alice' | 'Bob'

V -> 'loves'

""")

parser = nltk.ChartParser(grammar)

trees = parser.parse_all(text15)

for tree in trees:

    print(tree)
text16 = nltk.word_tokenize("I saw the man with a telescope")

grammar1 = nltk.CFG.fromstring("""

S -> NP VP

VP -> V NP | VP PP

PP -> P NP

NP -> DT N | DT N PP | 'I'

DT -> 'a' | 'the'

N -> 'man' | 'telescope'

V -> 'saw'

P -> 'with'

""")

parser1 = nltk.ChartParser(grammar1)

trees1 = parser1.parse_all(text16)

for tree in trees1:

    print(tree)
from nltk.corpus import treebank
text17 = treebank.parsed_sents('wsj_0001.mrg')[0]

print(text17)