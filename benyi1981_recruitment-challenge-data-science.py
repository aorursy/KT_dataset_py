import os
print(os.listdir("../input"))
!head "../input/amazon_cells_labelled.txt"
import numpy as np
import pandas as pd
amazon = pd.read_csv('../input/amazon_cells_labelled.txt',sep='\t',names=['text','label'])
yelp = pd.read_csv('../input/yelp_labelled.txt',sep='\t',names=['text','label'])
test = pd.read_csv('../input/imdb_labelled.txt',sep='\t',names=['text','label'])
#amazon.head()
#amazon.info()
#yelp.head()
#yelp.info()
#test.head()
#test.info()
train = pd.concat([amazon,yelp],axis=0)
#train.head()
#train.info()
#Q2 soln.

import string
from sklearn.feature_extraction.text import CountVectorizer

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split()]

bow = CountVectorizer(analyzer = text_process,stop_words='english').fit(train['text'])

#bow.get_shape()
#Q2a soln.
cv = CountVectorizer().fit(train['text'])
bag_of_words = cv.transform(train['text'])
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
sorted(words_freq, key = lambda x: x[1], reverse=True)
#Q3 soln.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.model_selection import train_test_split -- using imdb as test dasta
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow',CountVectorizer()) #analyzer=text_process, stop_words='english'
    ,('tfidf',TfidfTransformer())  
    ,('classifier',MultinomialNB())
])
pipeline.fit(train['text'],train['label'])
#Q4 soln.
predictions = pipeline.predict(test['text'])

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test['label'],predictions))
print('\n')
print(classification_report(test['label'],predictions))


## Overall, the model performs reasonably well with overall f1-score 0.74
## Recall is on the low side - instance of false negative appears to be high
## Problems: inherent difference using Amazon / Yelp reviews to predict IMDB reviews. Were they the same structure? are they the same types of audience?
## e.g. IMDB ratings go 1-10; Yelp 1-5, Amazon 1-5
## They are also of very different categories (restaurants vs movies vs everything), the textual description might differ significantly between categories
## Are they coded into binary decisions correctly?

## Possible extensions to this problem if required:
## 1. using train_test_split to split and train amazon/yelp data first before 
## 2. try other classification algorithms (e.g. RandomForest)
## 3. designing different features (I experimented with stopwords and analyzer, there are many other ways to do feature engineering)
