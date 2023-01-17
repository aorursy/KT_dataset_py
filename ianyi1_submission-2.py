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
import pandas as pd
import numpy as np
import matplotlib as mtpl
import matplotlib.pyplot as plt
import seaborn as sns 
import nltk
import os
import sys
import json
import requests
import pandas as pd
from nltk import clean_html
from bs4 import BeautifulSoup as bs
from textblob import TextBlob, Word
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
from textblob import Blobber
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
tokenizer = RegexpTokenizer('\w+')
df = pd.read_csv('../input/train.tsv', sep='\t', header=0, nrows = 5000)
df['Tokenized Phrase'] = 0

for i in range(len(df)):
    df['Tokenized Phrase'][i] = tokenizer.tokenize(df['Phrase'][i])
test_df = pd.read_csv('../input/test.tsv', sep='\t', header=0)
stopwords = stopwords.words('english')

all_words = []

for phrase in df['Tokenized Phrase']:
    all_words += phrase
    
# we dont want stopwords
words_without_stop = []

for w in all_words:
    if w not in stopwords:
        words_without_stop.append(w)
words_freqdist = nltk.FreqDist(w.lower() for w in words_without_stop)

words_freqdist.most_common(15)

word_features = list(words_freqdist.keys())[:200]
# phrase_contains function - will return if the phrase contains a word in word_features (list of most common words)

def phrase_contains(phrase):
    phrase_words = set(phrase)
    features = {}
    for word in word_features:
        features['contains(%s)' %word] = (word in phrase_words)
    return features

featuresets2 = []

# creating features from the phrase_contains function

for idx, phrase in enumerate(df['Tokenized Phrase']):
    if phrase != []:
    
        featuresets2.append((phrase_contains(phrase), df['Sentiment'][idx]))

        
# classifying on most common words

train_set, test_set = featuresets2[(len(featuresets2)/2):], featuresets2[:(len(featuresets2)/2)]

classifier2 = nltk.NaiveBayesClassifier.train(train_set)

nltk.classify.accuracy(classifier2,test_set)
postags = nltk.pos_tag(words_without_stop)
VB_words = []

for p in postags:
    if (p[1] == 'VB') or (p[1] == 'VBZ'):
        VB_words.append(p[0])
VBwords_freqdist = nltk.FreqDist(VB.lower() for VB in VB_words)
VBword_features = list(VBwords_freqdist.keys())[:200]
# phrase_contains function for VB words - will return if the phrase contains a word in word_features (list of most common words)

def phrase_contains_VB(phrase):
    phrase_words = set(phrase)
    features = {} 
    for word in VBword_features:
        features['contains(%s)' %word] = (word in phrase_words)
    return features

featuresets2 = []

for idx, phrase in enumerate(df['Tokenized Phrase']):
    if phrase != []:
    
        featuresets2.append((phrase_contains_VB(phrase), df['Sentiment'][idx]))

# classifying only on most common VB words        
        
train_set, test_set = featuresets2[2493:], featuresets2[:2493]

classifier2 = nltk.NaiveBayesClassifier.train(train_set)

nltk.classify.accuracy(classifier2,test_set)
sub_df = pd.read_csv('../input/sampleSubmission.csv')
sentiment = []

test_df['Tokenized Phrase'] = 0
test_df['Sentiment'] = 0

for i in range(len(test_df)):
    test_df['Tokenized Phrase'][i] = tokenizer.tokenize(test_df['Phrase'][i])
for i in range(len(test_df)):
    
    sub_df['Sentiment'][i] = classifier2.classify(phrase_contains(test_df['Tokenized Phrase'][i]))
sub_df
sub_df.to_csv(index = False)