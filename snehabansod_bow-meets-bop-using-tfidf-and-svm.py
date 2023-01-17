# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from bs4 import BeautifulSoup # text processing
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Read labeled training and test data

train =pd.read_csv('../input/labeledTrainData.tsv', delimiter='\t', quoting=3)
test= pd.read_csv('../input/testData.tsv',delimiter='\t', quoting=3)
# Any results you write to the current directory are saved as output.
# lemmatize
lemma=WordNetLemmatizer()
# Defining a function to preprocess and clean data:
#'BeautifulSoup package' used to clean the data removing unwanted HTML.
#'Re package' used to remove unwanted punctuations. Few punctuations like '!', '?' and numeric numbers 
#are not removed as it may be helpful in predicting sementics. 
#'Tokenizer' used to convert paragraph into array instead of split(). This has improved performace as
#it can treat puctuations as separate word. Further steps include 'Stemming' and getting rid of 'stopwords'


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review,"lxml").get_text()      # remove html
    letters = re.sub("[^a-zA-Z0-9!?'-]", " ", review_text)         # passing only alphabets, numbers and some few punctuations
    words_arr=[lemma.lemmatize(w) for w in word_tokenize(str(letters).lower())]   #Lammetize and tokenize
    stops = set(stopwords.words("english"))                                 
    meaningful_words = [w for w in words_arr if not w in stops]           #removing common english words
    return( " ".join( meaningful_words ))

## let's take one example and see the difference
train['review'][1]
# cleaned paragraph ex.

clean_review = review_to_words( train["review"][1] )
print(clean_review)
# we can proceed to process full training and test data
num_reviews = train["review"].size
clean_train_reviews = []

print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print ("Review %d of %d" % ( i+1, num_reviews ))                                                                   
    clean_train_reviews.append( review_to_words( train['review'][i] ))
# Cleaning and Parsing Test Data

numOfRev=len(test)
clean_test_reviews=[]
print("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,numOfRev):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\t" % (i+1, numOfRev))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )
#creating bag of words

from sklearn.feature_extraction.text import CountVectorizer              #Importing Vectorizer
vectorizer= CountVectorizer(analyzer='word',max_features=2500)

train_data_features = vectorizer.fit_transform(clean_train_reviews)     #Vectorizing training Data
train_data_features = train_data_features.toarray() 

test_data_features = vectorizer.transform(clean_test_reviews)            #Vectorize Test Data
test_data_features = test_data_features.toarray()
# TFIDF and SVM Classifier

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(train_data_features)       #  TFIDF
messages_tfidf = tfidf_transformer.transform(train_data_features)
test_tfidf=tfidf_transformer.transform(test_data_features)

from sklearn.svm import SVC, LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(messages_tfidf, train['sentiment'])                  # SVM
pred = linear_svc.predict(test_tfidf)
# test the accuracy
acc_linear_svc = round(linear_svc.score(messages_tfidf, train['sentiment']) * 100, 2)
acc_linear_svc
final_result = pd.DataFrame( data={"id":test["id"], "sentiment":pred})
final_result.to_csv('output', index=False, quoting=3)
