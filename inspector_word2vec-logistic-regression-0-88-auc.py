import nltk
from nltk.corpus import stopwords 
stops = set(stopwords.words("english"))
from gensim.models import Word2Vec
from bs4 import BeautifulSoup 
import re
import sqlite3
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import linear_model

#location of sqlite
dataLoc= '../input/database.sqlite'

pd.set_option('max_colwidth', 100)
def binarize_score(score):
    """
    set scores of 1-3 to 0 and 4-5 as 1
    """
    
    if score <3:
        return 0
    else:
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



connection = sqlite3.connect(dataLoc)
reviews = pd.read_sql_query(""" SELECT Score, Summary, Text FROM Reviews WHERE Score != 3 """, connection)

   
reviews['Score_binary']=reviews['Score'].apply(binarize_score)
reviews['word_list']=reviews['Summary'].apply(review_to_words)


print (reviews.head(n=10))

X_train, X_test, y_train, y_test = train_test_split(reviews['word_list'], reviews['Score_binary'], test_size=0.2, random_state=42)


#size of hidden layer (length of continuous word representation)
dimsize=400

#train word2vec on 80% of training data
model = Word2Vec(X_train.values, size=dimsize, window=5, min_count=5, workers=4)

#create average vector for train and test from model
#returned list of numpy arrays are then stacked 
X_train=np.concatenate([avg_word_vectors(w,dimsize) for w in X_train])
X_test=np.concatenate([avg_word_vectors(w,dimsize) for w in X_test])
#basic logistic regression with SGD
clf = linear_model.SGDClassifier(loss='log')
clf.fit(X_train, y_train)
p=clf.predict_proba(X_test)
roc_auc_score(y_test,p[:,1])