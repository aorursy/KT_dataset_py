import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
import re
import logging
import time
import warnings
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

import nltk
from nltk.corpus import stopwords 
train = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
sw = stopwords.words('english')
for w in ['film','see','get','make','fill']:
    sw.append(w)
sw = set(sw)
def stage_one(text):
    text = BeautifulSoup(text).get_text()   
    reg = re.sub("[^a-zA-Z]", " ", text) 
    low = reg.lower().split()   
    meaningful_words = [w for w in low if not w in sw]
    return( " ".join( meaningful_words ))
clean_train = []
for i in range( 0, train["review"].size ):                                                          
    clean_train.append( stage_one( train["review"][i] ))
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 12000, \
                             ngram_range=(1, 3)) 
features = vectorizer.fit_transform(clean_train).toarray()

dist = np.sum(features, axis=0)
vocab = vectorizer.get_feature_names()
    
forest = ExtraTreesClassifier(n_estimators = 150, n_jobs = -1) 
forest = forest.fit( features, train["sentiment"] )

test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", quoting=3 )
clean_test = [] 
for i in range(0,len(test["review"])):
    clean_review = stage_one( test["review"][i] )
    clean_test.append( clean_review )

test_features = vectorizer.transform(clean_test)
test_features = test_features.toarray()

result = forest.predict(test_features)

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv("n_vectorize_1_3 submit.csv", index=False, quoting=3 )


