import nltk 

from nltk.corpus import stopwords

from nltk.cluster.util import cosine_distance

import numpy as np

import networkx as nx

import pandas as pd 



    

import collections



import random

import re

from collections import Counter

from itertools import islice

import nltk

from nltk.corpus import stopwords

import numpy as np 

import pandas as pd 

pd.set_option('display.max_colwidth', -1)

from time import time

import re

import string

import os

import emoji

from pprint import pprint

import collections

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

sns.set(font_scale=1.3)

from sklearn.metrics import f1_score

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

import joblib

import gensim

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize

import warnings

warnings.filterwarnings('ignore')

np.random.seed(37)  

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder                                              
arabic_diacritics = re.compile(""" ّ    | # Tashdid

                             َ    | # Fatha

                             ً    | # Tanwin Fath

                             ُ    | # Damma

                             ٌ    | # Tanwin Damm

                             ِ    | # Kasra

                             ٍ    | # Tanwin Kasr

                             ْ    | # Sukun

                             ـ     # Tatwil/Kashida

                         """, re.VERBOSE)





def remove_diacritics(text):

    text = re.sub(arabic_diacritics, '', str(text))

    return text







def remove_repeating_char(text):

    # return re.sub(r'(.)\1+', r'\1', text)     # keep only 1 repeat

    return re.sub(r'(.)\1+', r'\1\1', text)  # keep 2 repeat



def remove_short_words(text): 

    return str(text).replace(r'\b(\w{1,3})\b', '')



def process_text(text, grams=False):

    clean_text = remove_diacritics(text)

    clean_text = remove_repeating_char(clean_text)

    clean_text = remove_short_words(clean_text)

    if grams is False:

        return clean_text.split()

    else:

        tokens = clean_text.split()

        grams = list(window(tokens))

        grams = [' '.join(g) for g in grams]

        grams = grams + tokens

        return grams





def window(words_seq, n=2):

    "Returns a sliding window (of width n) over data from the iterable"

    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "

    it = iter(words_seq)

    result = tuple(islice(it, n))

    if len(result) == n:

        yield result

    for elem in it:

        result = result[1:] + (elem,)

        yield result

def document_features(document, corpus_features):

    document_words = set(document)

    features = {}

    for word in corpus_features:

        features['contains({})'.format(word)] = (word in document_words)

    return features



def process_text1(text, grams=False):

    clean_text = remove_diacritics(text)

    clean_text = remove_repeating_char(clean_text)

    

    if grams is False:

        return nltk.tokenize.sent_tokenize(clean_text)

    else:

        tokens = nltk.tokenize.sent_tokenize(clean_text)

        grams = list(window(tokens))

        grams = [' '.join(g) for g in grams]

        grams = grams + tokens

        return grams



sentence_list = list() 

all_features = list()

texts = list()

data_labels = list()

#import pyarabic.arabrepr

#arepr = pyarabic.arabrepr.ArabicRepr()

#repr = arepr.repr

#from tashaphyne.stemming import ArabicLightStemmer

#ArListem = ArabicLightStemmer()



negative_file = open("../input/ts-naim-mhedhbi/negative_tweets.txt", encoding ="utf8")

positive_file = open("../input/ts-naim-mhedhbi/positive_tweets.txt", encoding ="utf8")

negative_file1 = open("../input/ts-naim-mhedhbi/negative_tweets.txt", encoding ="utf8")

positive_file1 = open("../input/ts-naim-mhedhbi/positive_tweets.txt", encoding ="utf8")







n_grams_flag = False

min_freq = 13



print('read data ...')

print('read data ...')

# read positive data

 
for line in positive_file:

    

    text_features = process_text(line, grams=n_grams_flag)

    stop_words = set(stopwords.words('arabic'))

    text_features = [w for w in text_features if not w in stop_words]

    all_features += text_features

    texts.append(text_features)

    data_labels.append('pos')   





for line in negative_file:

    

    text_features = process_text(line, grams=n_grams_flag)

    stop_words = set(stopwords.words('arabic'))

    text_features = [w for w in text_features if not w in stop_words]

    all_features += text_features

    texts.append(text_features)

    data_labels.append('neg')
for line in positive_file1:

    

    text_features = process_text1(line, grams=n_grams_flag)

    stop_words = set(stopwords.words('arabic'))

    text_features = [w for w in text_features if not w in stop_words]

    sentence_list += text_features

    texts.append(text_features)

    data_labels.append('pos')   





for line in negative_file1:

    

    text_features = process_text1(line, grams=n_grams_flag)

    stop_words = set(stopwords.words('arabic'))

    text_features = [w for w in text_features if not w in stop_words]

    sentence_list += text_features

    texts.append(text_features)

    data_labels.append('neg')
print('data size', len(data_labels))

print('# of positive', data_labels.count('pos'))

print('# of negative', data_labels.count('neg'))  

  

tweets = [(t, l) for t, l in zip(texts, data_labels)]



random.shuffle(tweets)

print('sample tweets')

for t in tweets[:10]: print(t)  # see the first 10 instances

print('all words sample')

print(all_features[:20])

print('len(all_words):', len(all_features))

all_features_freq = nltk.FreqDist(w for w in all_features)

print(all_features_freq)

print('sample frequencies')

print(all_features_freq.most_common(20))

print('freq of في', all_features_freq.freq('في'))

print('features frequencies are computed')

thr = min_freq / len(all_features)

print('selecting features')



#Finally, to find the weighted frequency, we can simply divide the number of occurances of all the words by the frequency of the most occurring word, as shown below:

maximum_frequncy = max(all_features_freq.values())

for word in all_features_freq.keys():

    all_features_freq[word] = (all_features_freq[word]/maximum_frequncy) 



    

sentence_scores = {}

for sent in sentence_list:

    for word in nltk.word_tokenize(sent.lower()):

        if word in all_features_freq.keys():

            if len(sent.split(' ')) < 30:

                if sent not in sentence_scores.keys():

                    sentence_scores[sent] = all_features_freq[word]

                else:

                    sentence_scores[sent] += all_features_freq[word]



import heapq

summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)



summary = ' '.join(summary_sentences)

print(summary)