# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# This is the code for naive bayes and this is sometimes also called as stupid bayes algorithm
import nltk
import random

# corpus is mostly related to speech or reviews
from nltk.corpus import movie_reviews

my_documents1 = [(list(movie_reviews.words(fileid)),category) 
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]



# Shuffled the my_documents1
random.shuffle(my_documents1)

print(type(my_documents1))

print(len(my_documents1))

# Accessing the first index tuple which have a list of words and category
print(my_documents1[1])

print(my_documents1[2])

print(my_documents1[3])


my_all_words = []

# Storing all the words of into a list from movie_reviews
for word in movie_reviews.words():

    # converting everything into the lower case
    my_all_words.append(word.lower())

# It gets the frequency of all the words....
my_all_words_freq = nltk.FreqDist(my_all_words)

type(my_all_words_freq)
len(my_all_words_freq)
# This will print the ddictionary for all the words with their respective frequency from higher to lower ....

my_all_words_freq
print(my_all_words_freq.keys())
# Printing starting 3000 keys 

print(list(my_all_words_freq.keys())[:3000])
# Storing all the keys or say words from 0 to 3000 in a variable called my_word_features
# We took 3000 because there are lots of non word characters in the data hence,we wanted to have healthy 
 # datasets for the words....
my_word_features = list(my_all_words_freq.keys())[:3000]
# Defining a function to get the features
def get_features(documents):
    words = set(documents)
    features = {}
    for x in my_word_features:
        # filling our above empty dictionary with boolean values for the keys
        features[x] = (x in words)
    return features

print((get_features(movie_reviews.words('neg/cv000_29416.txt'))))
# my_documents1 is a list of tuples and in each tuple we have lots of word and then category....
# So, every time in each for loop iteration we are passing some words with category and the function get_features()
 # will return key word and its boolean value

my_feature_sets = [(get_features(rev),category) for (rev, category) in my_documents1]
len(my_feature_sets)
type(my_feature_sets)
my_feature_sets[1]
my_feature_sets[2]
my_feature_sets[3]
my_training_set = my_feature_sets[:1900]
my_testing_set = my_feature_sets[1900:]
# posterior = prior_occurence * likelihood / evidence

my_classsifier = nltk.NaiveBayesClassifier.train(my_training_set)
print("The accuracy of my classifier in percentage is: ", (nltk.classify.accuracy(my_classsifier, my_testing_set))*100)
my_classsifier.show_most_informative_features()
print("The accuracy of my classifier in percentage is: ", (nltk.classify.accuracy(my_classsifier, my_testing_set))*100)
# Getting top 12 most occurring words

my_classsifier.show_most_informative_features(12)
