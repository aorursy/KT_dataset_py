%matplotlib inline

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords 
stops = set(stopwords.words("english"))
from gensim.models import Word2Vec
from bs4 import BeautifulSoup 
import re
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import linear_model

#location of sqlite
dataloc = '../input/database.sqlite'
con = sqlite3.connect(dataloc)
reviews = pd.read_sql_query("""
SELECT HelpfulnessNumerator, HelpfulnessDenominator, Text, Score
FROM Reviews
WHERE HelpfulnessDenominator >= 5
""", con)
reviews['HScore'] = 100 * (1.0 * reviews['HelpfulnessNumerator']) / (1.0 * reviews['HelpfulnessDenominator'])
reviews[['HScore']] = reviews[['HScore']].astype('int')
print(len(reviews))
print(reviews.head())
def binarize_score(score):
    """
    set scores of 0-30 to 0 and 31-79 as 1 and 80-100 as 2
    """
    
    if score <=30:
        return 0
    elif score >= 80:
        return 2
    return 1



def review_to_words( review ):
    """
    Return a list of cleaned word tokens from the raw review
    
    """
        
    #Remove any HTML tags and convert to lower case
    review_text = BeautifulSoup(review).get_text().lower() 
    
    #Replace smiliey and frown faces, ! and ? with coded word SM{int} in case these are valuable
    review_text=re.sub("(:\))",r' SM1',review_text)
    review_text=re.sub("(:\()",r' SM2',review_text)
    review_text=re.sub("(!)",r' SM3',review_text)
    review_text=re.sub("(\?)",r' SM4',review_text)
    
    #keep 'not' and the next word as negation may be important
    review_text=re.sub(r"not\s\b(.*?)\b", r"not_\1", review_text)
    
    #keep letters and the coded words above, replace the rest with whitespace
    nonnumbers_only=re.sub("[^a-zA-Z\_(SM\d)]"," ",review_text)  
    
    #Split into individual words on whitespace
    words = nonnumbers_only.split()                             
    
    #Remove stop words
    words = [w for w in words if not w in stops]   
    
    return (words)



def avg_word_vectors(wordlist,size):
    """
    returns a vector of zero for reviews containing words where none of them
    met the min_count or were not seen in the training set
    
    Otherwise return an average of the embeddings vectors
    
    """
    
    sumvec=np.zeros(shape=(1,size))
    wordcnt=0
    
    for w in wordlist:
        if w in model:
            sumvec += model[w]
            wordcnt +=1
    
    if wordcnt ==0:
        return sumvec
    
    else:
        return sumvec / wordcnt
reviews['HScore_binary']=reviews['HScore'].apply(binarize_score)
reviews['word_list']=reviews['Text'].apply(review_to_words)
print (reviews.head(n=10))
X_train, X_test, y_train, y_test = train_test_split(reviews['word_list'], reviews['HScore_binary'], test_size=0.2, random_state=42)

#size of hidden layer (length of continuous word representation)
dimsize=400

#train word2vec on 80% of training data
model = Word2Vec(X_train.values, size=dimsize, window=5, min_count=5, workers=4)

#create average vector for train and test from model
#returned list of numpy arrays are then stacked 
X_train=np.concatenate([avg_word_vectors(w,dimsize) for w in X_train])
X_test=np.concatenate([avg_word_vectors(w,dimsize) for w in X_test])
#basic logistic regression with SGD
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
clf = linear_model.SGDClassifier(loss='log')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
confusion_matrix(y_test, y_pred)
