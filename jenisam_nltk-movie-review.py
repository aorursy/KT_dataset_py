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
import nltk
from nltk.corpus import movie_reviews,stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
import string
from nltk import pos_tag
from nltk.stem import wordnet
movie_reviews.fileids()
len(movie_reviews.fileids())
movie_reviews.words(movie_reviews.fileids()[4])
movie_reviews.categories()
movie_reviews.fileids('pos')
movie_reviews.fileids('neg')
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
print(documents[1])
#to convert pos_tag systax to lemmatizer syntax
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.NOUN
stop = stopwords.words('english')
punc = list(string.punctuation)
stop = stop+punc
def clean(words):
    output=[]
    #print(words)
    #output of refined words
    for w in words:
        #if word is not in the stop list
         if w.lower() not in stop:
            #grab the pos_tag
            pos = pos_tag([w])[0][1]
            #get the root word
            clean= lemmatizer.lemmatize(w,pos=get_simple_pos(pos))
            #append it
            output.append(clean.lower())
    return output
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
clean(word_tokenize('i was playing'))
documents = [(clean(document),category) for document,category in documents]
