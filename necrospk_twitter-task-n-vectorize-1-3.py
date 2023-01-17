import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
import re
import logging
import time
import warnings
import gensim
import sys
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

from bs4 import BeautifulSoup 

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.cluster import KMeans

from gensim.models import Word2Vec 

from nltk.tokenize import WordPunctTokenizer, TweetTokenizer

import nltk
from nltk.corpus import stopwords 
train = pd.read_csv('../input/train.csv',encoding='latin1',sep=',')
test = pd.read_csv('../input/test.csv',encoding='latin1',sep=',')
stop_words = set(stopwords.words('english'))
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
def stage_one(text):
    text = BeautifulSoup(text).get_text()   
    reg = re.sub("[^a-zA-Z]", " ", text) 
    low = reg.lower().split()  
    meaningful_words = ''
    meaningful_words = tokenizer.tokenize(" ".join( low ))
    return( " ".join( meaningful_words ))
%%time
clean_train = []
for i in range( 0, train["SentimentText"].size ):                                                          
    if i % 10000==0:
        print(i,'samples is readt from',train["SentimentText"].size)
    clean_train.append( stage_one( train["SentimentText"][i] ))
%%time
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = stop_words,   \
                             max_features =1500, \
                             ngram_range=(1, 3)) 
features = vectorizer.fit_transform(clean_train).toarray()

dist = np.sum(features, axis=0)
vocab = vectorizer.get_feature_names()
    
forest = ExtraTreesClassifier(n_estimators = 200, n_jobs = -1) 
forest = forest.fit( features, train["Sentiment"] )
%%time
clean_test = [] 
for i in range(0,len(test["SentimentText"])):
    if i % 10000==0:
        print(i,'samples is readt from',train["SentimentText"].size)
    clean_review = stage_one( test["SentimentText"][i] )
    clean_test.append( clean_review )

test_features = vectorizer.transform(clean_test)
test_features = test_features.toarray()

result = forest.predict(test_features)

output = pd.DataFrame( data={"ItemID":test["ItemID"], "Sentiment":result} )
output.to_csv("n_vectorize_1_3 submit.csv", index=False, quoting=3 )

