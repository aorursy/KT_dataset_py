# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading the Trained Dataset
train = pd.read_csv("../input/labeledTrainData.tsv", header = 0, delimiter = '\t')
train.shape[0]

# Reading the first few rows of the Trained Dataset
train.head()
# Summary Statistics of the Trained Dataset
train.describe()
# Reading thr Test Dataset
test = pd.read_csv("../input/testData.tsv", header = 0, delimiter = '\t')
test.shape[0]
# Reading the first few rows of Trained Dataset
test.head()
# Summary statistics of Test Dataset
test.describe()
from nltk.corpus import stopwords
def rev_to_words(review):
        #Remove HTML
        rev_text = BeautifulSoup(review).get_text()
        #Removes Numbers/Non Letters
        letter_only = re.sub("[^a-zA-Z]", " ", rev_text)   
        #Converts the letters into lowercase and splits them.
        words = letter_only.lower().split() 
        #Search set of unique words in the Data
        stops = set(stopwords.words("english"))
        #Remove stop words
        meaningful_words = [n for n in words if not n in stops]  
        #joining of meaningful words together
        return(" ".join( meaningful_words ))
num_reviews = train["review"].size
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( rev_to_words( train["review"][i] ) )
from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
from sklearn.ensemble import RandomForestClassifier
#Applying Random forest Classifier method with 150 trees
forest = RandomForestClassifier(n_estimators = 150)
#Fit the forest to the training dataset
forest = forest.fit( train_data_features, train["sentiment"] )
#Create an empty list in the test and append the clear reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

for i in range(0,num_reviews):
    clean_review = rev_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
#Result and Output 
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

output.to_csv( "submit.csv", index=False, quoting=3 )

