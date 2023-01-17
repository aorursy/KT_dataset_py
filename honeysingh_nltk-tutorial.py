# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import matplotlib

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
sentence = "android.txt"
stop_words = set(stopwords.words("english"))
filtered =[]
words = word_tokenize(sentence)
for w in words:
    if w not in stop_words:
        filtered.append(w)
filtered_words = [w for w in words if not w in stop_words]
filtered_words
from nltk.stem import PorterStemmer
ps = PorterStemmer()
words = ["enable","enabling","enabled"]
for w in words:
    print(ps.stem(w))
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
#input the text file from corpus state_union
sample = state_union.raw("2005-GWBush.txt")
#create a punkt tokenizer 
#if we give any sample files, it'll be trainied ussing that text
#in this scenario, it is pretrained
tokenizer = PunktSentenceTokenizer()
sentence = tokenizer.tokenize(sample)
def process_content():
    try:
        for w in sentence[:1]:
            words = word_tokenize(w)
            #word and its tag is given
            tags = nltk.pos_tag(words)
            print(tags)
    except Exception as e:
        print(str(e))

process_content()
custom_sent_tokenizer = PunktSentenceTokenizer()

tokenized = custom_sent_tokenizer.tokenize(sample)

def process_contents():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked)   

    except Exception as e:
        print(str(e))

process_contents()
#Chinking is a lot like chunking, it is basically a way for you to remove a chunk from a chunk
def process_chinking():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{""" #chinking one or more verbs, prepositions, determiners, or the word 'to'.

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            print(chunked)

    except Exception as e:
        print(str(e))

process_chinking()
#able to pull out "entities" like people, places, things, locations, monetary figures, and more.
tokenized = custom_sent_tokenizer.tokenize(sample)

def process_NER():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged)
            print(namedEnt)
    except Exception as e:
        print(str(e))


process_NER()
#A very similar operation to stemming is called lemmatizing. The major difference between these is, as you saw earlier,
#stemming can often create non-existent words, whereas lemmas are actual words.

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))
from nltk.corpus import wordnet
syns = wordnet.synsets("love")
#print(syns)
#print(syns[0])
#print(syns[0].name())
print(syns[0].lemmas())
print(syns[0].definition())

print(syns[0].examples())
#Word similarity
w1 = wordnet.synset("savior.n.01")
w2 = wordnet.synset("jesus.n.01")

print(w1.wup_similarity(w2))
import random
from nltk.corpus import movie_reviews
documents = [((list(movie_reviews.words(fileid)), category))
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

print(documents[1])
random.shuffle(documents)
print(documents[1])
all_words =[]
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
print(all_words["silly"])
word_features  = list(all_words.keys())[:3000]
def features(document):
    words  = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features
featureset = [(features(document),category) for (document,category) in documents]
import sentiment_mod as s
print(s.sentiment("It was too good"))
