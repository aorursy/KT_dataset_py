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
#importing the training data 

train = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', delimiter="\t", header=0, quoting=3)
train.shape

train.columns.values
train.shape
print (train['review'][0])
#Data Cleaning and Text Processing

#Removing HTML tags and markups

from bs4 import BeautifulSoup
example1=BeautifulSoup(train['review'][0])
#printing the raw review and the output for comparison

print(train['review'][0])

print(example1.get_text())
#Removing the Punctuations and Numbers

import re
#USe regular expressions to find and replace

letters_only=re.sub("[^a-zA-Z]", " ", example1.get_text())
print (letters_only)
#Converting it to small letters 

lower_case=letters_only.lower()



#Splitting it into words

words=lower_case.split()

print (words)
#Dealing with stopwords

import nltk

nltk.download()
from nltk.corpus import stopwords

#remove stopwords from the movie review

words = [w for w in words if not w in stopwords.words("english")]
print (words)
def review_to_words(raw_review):

    #1.Remove HTML tags and markups

    review_text=BeautifulSoup(raw_review).get_text()

    #2.Remove non letters

    letters_only=re.sub("[^a-zA-Z]", " ", review_text)

    #3.Convert it to lowercase and split it into words

    words=letters_only.lower().split()

    #4.Converting stopwords list into sets

    stops=set(stopwords.words('english'))

    #5.Removing stopwords from reviews

    meaningful_words=[w for w in words if not w in stopwords.words('english')]

    #Joining the words back into string

    return (" ".join(meaningful_words))
#get the length of training reviews

num_reviews=len(train)
#Initialising an empty list for storing cleaned reviews

clean_train_reviews1=[]
print ("Cleaning and parsing the training set movie reviews...\n")

for i in range( 0, num_reviews ):

    # If the index is evenly divisible by 1000, print a message

    if( (i+1)%1000 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_reviews ))                                                                    

    clean_train_reviews1.append( review_to_words( train["review"][i] ))
print ("Creating Bag of words Model")

from sklearn.feature_extraction.text import CountVectorizer
vectoriser=CountVectorizer(analyzer='word',

                          tokenizer=None,

                          preprocessor=None,

                          stop_words=None,

                          max_features=5000)
train_data_features=vectoriser.fit_transform(clean_train_reviews1)
train_data_features=train_data_features.toarray()

print (train_data_features.shape)
#random forest classifier

print("Training the Random forest Classifier")

from sklearn.ensemble import RandomForestClassifier
#initialising Random forest classifier with 100 classifiers

classifier=RandomForestClassifier(n_estimators=100)
#Fitting the bag of words training model to the classifier

classifier=classifier.fit(train_data_features, train['sentiment'])
#Creating the Submission

#Reading the test file

test=pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter='\t', quoting=3)
#Creating an empty list and append the clean reviews

num_reviews=len(test["review"])

clean_test_data1=[]



print("Cleaning and Parsing the Test set data..")

for i in range(0, num_reviews):

    if( (i+1)%5000 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_reviews )) 

    clean_test_data1.append(review_to_words(test['review'][i]))

    
#Creating the Bag of words model for the test data

test_data_features=vectoriser.transform(clean_test_data)

test_data_features=test_data_features.toarray()
#Making prediction for the test dataset

result=classifier.predict(test_data_features)
# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )



# Use pandas to write the comma-separated output file

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )