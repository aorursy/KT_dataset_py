# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from bs4 import BeautifulSoup
import nltk
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
import os
import time
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/labeledTrainData.tsv", header = 0, delimiter = '\t')

def review_to_words( raw_review):
        review_text = BeautifulSoup(raw_review, features="html5lib").get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)            
        words = letters_only.lower().split() 
        stops = set(stopwords.words("english"))                         
        meaningful_words = [w for w in words if not w in stops]
        element = " ".join( meaningful_words )     
        # print(element)   
        return(element)

num_reviews = train["review"].size
clean_train_reviews = []

def fn(raw_review):
        element = review_to_words(raw_review)
        return(element)
        
print("Cleaning the Reviews...")
pool = Pool()
clean_train_reviews = pool.map(fn,train["review"])
pool.close()
pool.join()

print("The length of the Cleaned Reviews is "+str(len(clean_train_reviews)))
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 1500) 
train_data_features = vectorizer.fit_transform(clean_train_reviews)
print("Creating Features...")
train_data_features = train_data_features.toarray()
print("Features Created")
from sklearn.svm import SVC
print("Going to Train the Data Features")
print("The length of the array is "+str(len(train_data_features)))
clf = SVC(gamma='auto',cache_size=12000,max_iter=-1)
print("Training the data set...")
clf = clf.fit(train_data_features, train["sentiment"])
print("Training Completed")
test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", \
                    quoting=3 )
print("Reading the Test Data...")
num_reviews = len(test["review"])
clean_test_reviews = [] 
print("Total number of reviews is "+str(num_reviews))
pool = Pool()
clean_test_reviews = pool.map(fn,test["review"])
pool.close()
pool.join()
print("Cleaning of the Test Reviews is Completed!!")
print("The length of the cleaned test reviews is "+str(len(clean_test_reviews)))

test_data_features = vectorizer.transform(clean_test_reviews)
print("Making the features from the Test Reviews...")
test_data_features = test_data_features.toarray()
print("Going to predict the Test Features...")
result = clf.predict(test_data_features)
print("Prediction Completed!!")
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
print("Saving the Predictions to the File")
output.to_csv( "results.csv", index=False, quoting=3 )