import pandas as pd

path="../input/CORD-19-research-challenge/2020-03-13/"

all_sources=pd.read_csv(path+"all_sources_metadata_2020-03-13.csv")



all_sources.head()
import nltk

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



corpus = []

corpus = all_sources['title'].dropna()

corpus
from nltk.probability import FreqDist

import numpy as np



lemmatizer = WordNetLemmatizer()

stop_word = set(stopwords.words("english"))

symbol = ['(',')',':','!','?',',','.']



lem_word=[]



for x in corpus:

    lem_word.append(lemmatizer.lemmatize(x.lower()))



tokens_title = []



for i in lem_word:

    tokens_title+=word_tokenize(i)



stopwords_title = [z for z in tokens_title if z not in stop_word and z not in symbol]

freq_dist_title = FreqDist(stopwords_title)

freq_dist_title.most_common(10)
X,Y = zip(*freq_dist_title.most_common(10))
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(15,7))

plt.bar(X,Y)
def top_common_word(pd_data,col='abstract'):

    global stopwords

    global tokens

    stop_wrd = set(stopwords.words("english"))

    symbol = ['(',')',':','!','?',',','.','%','@']

    new_corpus = pd_data[col].dropna()

    lemmatizer = WordNetLemmatizer()

    

    lem_wrd = []

    

    for x in new_corpus:

        lem_wrd.append(lemmatizer.lemmatize(x.lower()))

    

    tokens = []

    

    for i in lem_wrd:

        tokens+=word_tokenize(i)



    stopwords = [z for z in tokens if z not in stop_wrd and z not in symbol]

    

    freq_dist = FreqDist(stopwords)

    

    return freq_dist

    
freq_dist_abstract = []

freq_dist_abstract = top_common_word(all_sources,'abstract')
X,Y = zip(*freq_dist_abstract.most_common(10))

plt.figure(figsize=(15,7))

plt.bar(X,Y)