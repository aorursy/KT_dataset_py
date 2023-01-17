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
data = pd.read_table('../input/labeledTrainData.tsv')
data.head()
data.info()
data['review'][0]
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import word2vec

stemmer = PorterStemmer()
lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')
import re

def clean_review(review):
    clean_words = []
    
    review = stemmer.stem(review)
    review = lemma.lemmatize(review)
    
    words = word_tokenize(review)
    for word in words:
        word = word.lower()
        word = re.sub("[0-9]|(\.|\/|,|;|@|#|\$|\&|\^|\*|\(|\)|-|_|\+|=|\?|!|)", "", word)
        if word not in stop_words:
            clean_words.append(word)
    return clean_words
rv1 = data['review'][0]
# type(rv1)
# words = clean_review(rv1)
# print(words)
data['review_cleaned'] = data['review'].apply(clean_review)
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
num_features = 300  # Word vector dimensionality
min_word_count = 40 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words


model = word2vec.Word2Vec(data['review_cleaned'],\
                          workers=num_workers,\
                          size=num_features,\
                          min_count=min_word_count,\
                          window=context,
                          sample=downsampling)

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "imdb_word2vec"
model.save(model_name)
model.wv.doesnt_match("france england germany washington".split())
model.wv.most_similar("plane")
