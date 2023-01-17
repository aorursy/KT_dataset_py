# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk # tool kit for natural language processing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# String to process for start 

string = "NLP is building systems that can understand language. It is subset of Artificial Intelligence"

from nltk.tokenize import word_tokenize, sent_tokenize 

sentence = sent_tokenize(string) # sentence tockenization 

print(sentence)
word = word_tokenize(string) # word tockenization

print(word)
from nltk.corpus import stopwords

from string import punctuation

stopwordsSet = set(stopwords.words('english')+list(punctuation)) # setting english words and list of punctuation

wordsStopwords = [words for words in word_tokenize(string) if words not in stopwordsSet]

# words without stopwords

print(wordsStopwords)
from nltk.collocations import *

string2 = "NLP is building systems that can understand language language language. It is subset of Artificial Intelligence"

# taking second string with repeated words

wordsStopwords2 = [words for words in word_tokenize(string2) if words not in stopwordsSet] # words without stopwords

words_bigram = nltk.collocations.BigramAssocMeasures() #measurement of bigrams

words_finder = nltk.BigramCollocationFinder.from_words(wordsStopwords2) # finding bigrams

sorted(words_finder.ngram_fd.items()) # printing found bigram items
string3 = "People live close when they work closely but they are closed enough"

# third string with repeated words of same meaning, so we can stem those 

from nltk.stem.lancaster import LancasterStemmer

lst = LancasterStemmer() # initiating LancasterStemmer

words_stemm = [lst.stem(word) for word in word_tokenize(string3)] # stemming similar words 

print(words_stemm) 
nltk.pos_tag(word_tokenize(string3)) # part of speech tagging, see output below, words are verb, nouns etc.
from nltk.corpus import wordnet as wn

for ss in wn.synsets('bass'):

    print(ss, ss.definition()) # printing all possible meaning of word bass in python built in therausus

from nltk.wsd import lesk

sense1 = lesk(word_tokenize("Sing in a lower tone, along with the bass"),'bass') 

print(sense1, sense1.definition()) # showing single possible meaning of above sentence with respect to bass word        
sense2 = lesk(word_tokenize("This sea bass was really hard to catch"),'bass')

print(sense2, sense2.definition()) # showing single possible meaning of above sentence with respect to bass word
import urllib.request

from bs4 import BeautifulSoup

#https://www.washingtonpost.com/news/the-switch/wp/2016/10/18/the-pentagons-massive-new-telescope-is-designed-to-track-space-junk-and-watch-out-for-killer-asteroids/

articleURL = "http://extension.nirsoft.net/wss"

with urllib.request.urlopen(articleURL) as response:

    page = response.read()

#page

#soup = BeautifulSoup(page,"lxml")