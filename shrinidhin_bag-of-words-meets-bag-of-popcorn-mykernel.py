# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import zipfile

with zipfile.ZipFile('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', 'r') as zip_ref:

    zip_ref.extractall('../output')
train_data=pd.read_csv('../output/labeledTrainData.tsv',delimiter='\t')
train_data.head()
train_data.shape
print(train_data['review'][0])
#We will use BeautifulSoup for pre-processing the reviews text

from bs4 import BeautifulSoup

import re

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
#This function performs all pre-processing required for a movie review

def pre_process_reviews(review):

    #Removing HTML Markup Text using BeautifulSoup Library

    obj=BeautifulSoup(review)

    

    #Using Regular Expressions to further process the string

    process = re.sub("[^a-zA-Z?!.,-/(/);:]", # The pattern to search for

                      " ",                   # The pattern to replace it with

                      obj.get_text())  # The text to search

    #We will convert the string to lowercase letter and divide them into words

    words=process.lower().split()

    

    #Searching a set is much faster than searching list, so we will convert the stop words into a set

    stops = set(stopwords.words("english")) 

    

    #We now remove the stop words or the unimportant words and retain only meaningful ones

    mean_words=[w for w in words if not w in stops]

    

    #Join final set of words into a meaningful string

    return " ".join(mean_words)
#We will remove the markup text from all the reviews and keep only the raw text for model building

train_data['review']=[pre_process_reviews(review) for review in train_data['review']]
#Initializing the CountVectorizer object for implementing Bag Of Words

vector=CountVectorizer(analyzer='word',

                       tokenizer=None,

                      preprocessor=None,

                      stop_words=None,

                      max_features=5000)

#Fitting and transforming into feature vectors

train_data_features=vector.fit_transform(train_data['review'])
vector.get_feature_names()
from sklearn.linear_model import LogisticRegression



logis=LogisticRegression()

logis.fit(train_data_features,train_data['sentiment'])
#Reading the test data

with zipfile.ZipFile('../input/word2vec-nlp-tutorial/testData.tsv.zip', 'r') as zip_ref:

    zip_ref.extractall('../output')

test_data=pd.read_csv('../output/testData.tsv',delimiter='\t')
test_data.head()
#Pre-processing the test data

test_data['review']=[pre_process_reviews(review) for review in test_data['review']]
# Get a bag of words for the test set, and convert to a numpy array

test_data_features = vector.transform(test_data['review'])

test_data_features = test_data_features.toarray()
#Use Log Regression model to predict sentiment

results=logis.predict(test_data_features)
#Preparing submission file

output=pd.DataFrame(data={'ID':test_data['id'],'sentiment':results})

output.to_csv('mysubmission.csv',index=False)