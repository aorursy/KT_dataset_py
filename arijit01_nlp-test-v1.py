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
#Getting the data into dataframes

#reviews_df = pd.read_csv("../input/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")

reviewsMay19_df = pd.read_csv("../input/amazon-consumer-reviews-may19/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
print(reviewsMay19_df)
#Information and description about the data

reviewsMay19_df.info()

reviewsMay19_df.describe()
# Columns in the dataset

reviewsMay19_df.columns
columns2drop = ['imageURLs', 'reviews.sourceURLs', 'sourceURLs']

reviewsMay19_df.drop(columns2drop, axis=1, inplace=True)

reviewsMay19_df.columns
reviewsMay19_df.head()
reviewsMay19_df.name.unique()
df = reviewsMay19_df[['asins', 'name','reviews.username',  'reviews.rating','reviews.title','reviews.text']].copy()
df.head()
#Importing all

import nltk

import random

from nltk.classify.scikitlearn import SklearnClassifier

import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI

from statistics import mode

from nltk.tokenize import word_tokenize

import re
df.head()
df.tail()
pos_df = df.loc[df['reviews.rating'] > 3]

pos_df.head()
neg_df = df.loc[df['reviews.rating'] <= 3]

neg_df.head()
print(len(pos_df))

print(len(neg_df))
#Testing for only 1 Review

all_words = []

documents = []



from nltk.corpus import stopwords



import re



stop_words = list(set(stopwords.words('english')))



#j is adject, r is adverb, and v is verb

allowed_word_types = ["J"]



pos_df1 = pos_df.head(1)

neg_df1 = neg_df.head(1)
pos_df1['reviews.title']
#Start cleaning and POS tagging the information



#Create a list of tuples where the first element of each tuple is a review and the second is the label

p = pos_df.iloc[0][5]

documents.append((p, "pos"))

documents
#Remove punctuations

cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

cleaned
#tokenize

tokenize = word_tokenize(cleaned)

tokenize
#Remove stopwords

stopped = [w for w in tokenize if not w in stop_words]

stopped
#Parts of speech tagging for every word

pos = nltk.pos_tag(stopped)

pos
# make a list of all the adjectives identified by the allowed word type list above

for w in pos:

    if w[1][0] in allowed_word_types:

        all_words.append(w[0].lower())



all_words
#Executing the cleaning and the POS tagging for all the reviews (positive and negative)

#Setup code again

all_words = []

documents = []



from nltk.corpus import stopwords



import re



stop_words = list(set(stopwords.words('english')))



#j is adject, r is adverb, and v is verb

allowed_word_types = ["J"]

pos_reviews = pos_df['reviews.title'].tolist()

neg_reviews = neg_df['reviews.title'].tolist()



for p in pos_reviews:

    #Create a list of tuples where the first element of each tuple is a review and the second is the label

    documents.append((p, "pos"))

    

    #Remove punctuations

    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

    

    #tokenize

    tokenize = word_tokenize(cleaned)

    

    #Remove stopwords

    stopped = [w for w in tokenize if not w in stop_words]

    

    #Parts of speech tagging for every word

    pos = nltk.pos_tag(stopped)

    

    # make a list of all the adjectives identified by the allowed word type list above

    for w in pos:

        if w[1][0] in allowed_word_types:

            all_words.append(w[0].lower())



for p in neg_reviews:

    #Create a list of tuples where the first element of each tuple is a review and the second is the label

    documents.append((p, "neg"))

    

    #Remove punctuations

    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

    

    #tokenize

    tokenize = word_tokenize(cleaned)

    

    #Remove stopwords

    stopped = [w for w in tokenize if not w in stop_words]

    

    #Parts of speech tagging for every word

    pos = nltk.pos_tag(stopped)

    

    # make a list of all the adjectives identified by the allowed word type list above

    for w in pos:

        if w[1][0] in allowed_word_types:

            all_words.append(w[0].lower())
documents
all_words
#Creating frequency distribution for each adjectives

all_words = nltk.FreqDist(all_words)



#list the 5000 most frequent words

word_features = list(all_words.keys())[:5000]



#function to create a dictionary of features for each review in the list document

#the keys are the words in word_features

#the values of each key are either true or false for whether that feature appears in the review



def find_features(document):

    words = word_tokenize(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)

    return features





#Creating features for each review

featuresets = [(find_features(rev), category) for (rev, category) in documents]



#Shuffling the documents

random.shuffle(featuresets)



training_set = featuresets[:20000]

testing_set = featuresets[20000:]
classifier = nltk.NaiveBayesClassifier.train(training_set)



print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)



classifier.show_most_informative_features(20)
classifier.show_most_informative_features(50)
from sklearn.model_selection import train_test_split
#Creating my own modifications to get numerical data instead of the True/False in feature sets

#Creating frequency distribution for each adjectives

all_words = nltk.FreqDist(all_words)



#list the 5000 most frequent words

word_features = list(all_words.keys())[:5000]



#function to create a dictionary of features for each review in the list document

#the keys are the words in word_features

#the values of each key are either true or false for whether that feature appears in the review



def find_features(document):

    words = word_tokenize(document)

    features = []

    for w in word_features:

        if w in words:

            features.append(1)

        else:

            features.append(0)

    return features



#Find the features for 1 row to test the features are in the format required

rev, category = documents[0]

print(find_features(rev))



#Creating features for each review

#featuresets = [(find_features(rev), category) for (rev, category) in documents]



#Shuffling the documents

#random.shuffle(featuresets)

#Creating the code to generate the data for all columns

#Creating my own modifications to get numerical data instead of the True/False in feature sets

#Creating frequency distribution for each adjectives

all_words = nltk.FreqDist(all_words)



#list the 5000 most frequent words

word_features = list(all_words.keys())[:5000]



#function to create a dictionary of features for each review in the list document

#the keys are the words in word_features

#the values of each key are either true or false for whether that feature appears in the review



def find_features(document):

    words = word_tokenize(document)

    features = []

    for w in word_features:

        if w in words:

            features.append(1)

        else:

            features.append(0)

    return features



dataset = []

result = []

for rev,category in documents:

    dataset.append(find_features(rev))

    if category == "pos":

        result.append(1)

    else:

        result.append(0)



print(len(dataset))

print(len(result))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(dataset, result, test_size=0.3, random_state=42)



print(len(X_train))

print(len(X_test))
from sklearn.svm import SVC



clf = SVC(gamma='auto')

clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score



predicted = clf.predict(X_test)

print(accuracy_score(y_test, predicted))