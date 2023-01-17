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
import nltk

import random

#from nltk.corpus import movie_reviews

from nltk.classify.scikitlearn import SklearnClassifier

import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI

from statistics import mode

from nltk.tokenize import word_tokenize



pos_fname = "/kaggle/input/shortreviews/sr_positive.txt"

neg_fname = "/kaggle/input/shortreviews/sr_negative.txt"



class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):

        self._classifiers = classifiers



    def classify(self, features):

        votes = []

        for c in self._classifiers:

            v = c.classify(features)

            votes.append(v)

        return mode(votes)



    def confidence(self, features):

        votes = []

        for c in self._classifiers:

            v = c.classify(features)

            votes.append(v)



        choice_votes = votes.count(mode(votes))

        conf = choice_votes / len(votes)

        return conf

    

short_pos = open('/kaggle/input/shortreviews/sr_positive.txt','r').read()

short_neg = open('/kaggle/input/shortreviews/sr_negative.txt','r').read()



all_words = []

documents = []





#  j is adject, r is adverb, and v is verb

#allowed_word_types = ["J","R","V"]

allowed_word_types = ["J"]



for p in short_pos.split('\n'):

    documents.append( (p, "pos") )

    words = word_tokenize(p)

    pos = nltk.pos_tag(words)

    for w in pos:

        if w[1][0] in allowed_word_types:

            all_words.append(w[0].lower())



    

for p in short_neg.split('\n'):

    documents.append( (p, "neg") )

    words = word_tokenize(p)

    pos = nltk.pos_tag(words)

    for w in pos:

        if w[1][0] in allowed_word_types:

            all_words.append(w[0].lower())







save_documents = open("pickled_algos/documents.pickle","wb")

pickle.dump(documents, save_documents)

save_documents.close()





all_words = nltk.FreqDist(all_words)

"""

FreqDist({'good': 369, 'more': 331, 'little': 265, ...})

"""



word_features = list(all_words.keys())[:5000]

"""

['21st', 'new', 'conan', 'greater', 'jean-claud',...]

"""



save_word_features = open("pickled_algos/word_features5k.pickle","wb")

pickle.dump(word_features, save_word_features)

save_word_features.close()





def find_features(document):

    words = word_tokenize(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)



    return features



featuresets = [(find_features(rev), category) for (rev, category) in documents]



save_featuresets = open("pickled_algos/featuresets.pickle", "wb")

pickle.dump(featuresets, save_featuresets)

save_featuresets.close()



random.shuffle(featuresets)

print(len(featuresets))



testing_set = featuresets[10000:]

training_set = featuresets[:10000]



#####

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)



save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")

pickle.dump(classifier, save_classifier)

save_classifier.close()



#####

MNB_classifier = SklearnClassifier(MultinomialNB())

MNB_classifier.train(training_set)

print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")

pickle.dump(MNB_classifier, save_classifier)

save_classifier.close()



#####

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())

BernoulliNB_classifier.train(training_set)

print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)



save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")

pickle.dump(BernoulliNB_classifier, save_classifier)

save_classifier.close()



#####

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())

LogisticRegression_classifier.train(training_set)

print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)



save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")

pickle.dump(LogisticRegression_classifier, save_classifier)

save_classifier.close()



#####

LinearSVC_classifier = SklearnClassifier(LinearSVC())

LinearSVC_classifier.train(training_set)

print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)



save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")

pickle.dump(LinearSVC_classifier, save_classifier)

save_classifier.close()





##NuSVC_classifier = SklearnClassifier(NuSVC())

##NuSVC_classifier.train(training_set)

##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)





SGDC_classifier = SklearnClassifier(SGDClassifier())

SGDC_classifier.train(training_set)

print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)



save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")

pickle.dump(SGDC_classifier, save_classifier)

save_classifier.close()

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







class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):

        self._classifiers = classifiers



    def classify(self, features):

        votes = []

        for c in self._classifiers:

            v = c.classify(features)

            votes.append(v)

        return mode(votes)



    def confidence(self, features):

        votes = []

        for c in self._classifiers:

            v = c.classify(features)

            votes.append(v)

        print('confidence :',votes)

        choice_votes = votes.count(mode(votes))

        conf = choice_votes / len(votes)

        return conf





documents_f = open("/kaggle/input/pickled-algos/documents.pickle", "rb")

documents = pickle.load(documents_f)

documents_f.close()



word_features5k_f = open("/kaggle/input/pickled-algos/word_features5k.pickle", "rb")

word_features = pickle.load(word_features5k_f)

word_features5k_f.close()





def find_features(document):

    words = word_tokenize(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)



    return features







featuresets_f = open("/kaggle/input/pickled-algos/featuresets.pickle", "rb")

featuresets = pickle.load(featuresets_f)

featuresets_f.close()



random.shuffle(featuresets)

print(len(featuresets))



testing_set = featuresets[10000:]

training_set = featuresets[:10000]







open_file = open("/kaggle/input/pickled-algos/originalnaivebayes5k.pickle", "rb")

classifier = pickle.load(open_file)

open_file.close()





open_file = open("/kaggle/input/pickled-algos/MNB_classifier5k.pickle", "rb")

MNB_classifier = pickle.load(open_file)

open_file.close()







open_file = open("/kaggle/input/pickled-algos/BernoulliNB_classifier5k.pickle", "rb")

BernoulliNB_classifier = pickle.load(open_file)

open_file.close()





open_file = open("/kaggle/input/pickled-algos/LogisticRegression_classifier5k.pickle", "rb")

LogisticRegression_classifier = pickle.load(open_file)

open_file.close()





open_file = open("/kaggle/input/pickled-algos/LinearSVC_classifier5k.pickle", "rb")

LinearSVC_classifier = pickle.load(open_file)

open_file.close()





open_file = open("/kaggle/input/pickled-algos/SGDC_classifier5k.pickle", "rb")

SGDC_classifier = pickle.load(open_file)

open_file.close()









voted_classifier = VoteClassifier(

                                  classifier,

                                  LinearSVC_classifier,

                                  MNB_classifier,

                                  BernoulliNB_classifier,

                                  LogisticRegression_classifier)









def sentiment(text):

    feats = find_features(text)

    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

print(sentiment("The jeans are awesome"))

print(sentiment("I won't recommend anyone to purchase from this seller"))

print(sentiment("I don't like this product"))

print(sentiment("amazing job by director"))
nRowsRead = 15 # specify 'None' if want to read whole file

# Womens Clothing E-Commerce Reviews.csv has 23486 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/review_ecommerce/Womens Clothing E-Commerce Reviews.csv', delimiter=',', nrows = nRowsRead, usecols = ['Review Text'])

df1.dataframeName = 'Womens Clothing E-Commerce Reviews.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1
df1.__dir__
df1['Sentiment'] = df1.apply(lambda row : sentiment(str(row['Review Text'])), axis = 1)     

print(df1)