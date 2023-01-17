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
from nltk.tokenize import word_tokenize as wtoken

# Import Counter

from collections import Counter as ct



# sentence = "The cat is in the box. The cat likes the box. The box is over the cat."

sentence = "The cat is in the box. The cat box."

print(sentence)



# Tokenize the sentence

print("{'Token': frequency of token}")

tokens = ct(wtoken(sentence))



# Convert the tokens into lowercase: lower_tokens

# otherwise 'The' and 'the' are considered as two separate entities.

lower_tokens = [t.lower() for t in tokens]



# Create a Counter with the lowercase tokens: bow_simple

bow_simple = ct(lower_tokens)



#Tracking the 5 most commonly occuring tokens

print(bow_simple.most_common(5))
from nltk.tokenize import word_tokenize as wtoken

# Import Counter

from collections import Counter as ct



# sentence = "The cat is in the box. The cat likes the box. The box is over the cat."

sentence = "The cat is in the box. The cat box."

print("sentence: ",sentence)



# Tokenize the sentence

tokens = wtoken(sentence)

print("tokenised sentence: ",tokens)



# Convert the tokens into lowercase: lower_tokens

# otherwise 'The' and 'the' are considered as two separate entities.

lower_tokens = [t.lower() for t in tokens]

print("lower case tokens: ",lower_tokens)



# Create a Counter with the lowercase tokens: bow_simple

bow_simple = ct(lower_tokens)

print("{'Token': frequency of token}: ",bow_simple)



#Tracking the 5 most commonly occuring tokens

print("5 most commonly occuring tokens: ",bow_simple.most_common(5))
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

print('StopWords :',stopWords)
# Text Preprocessing

"""

removal of stop words and non-alphabetic characters, lemmatize, and perform a new bag-of-words on your cleaned text.

"""

# Import WordNetLemmatizer

from nltk.stem import WordNetLemmatizer



sentence = "The cats are in the box. The cat likes the boxes. The box is over the cat."

lower_tokens = [t.lower() for t in wtoken(sentence)]

print('Initial list of tokens: ',lower_tokens)



# Create a list of only alphabetical characters. discard numerical-redundant values

# Retain alphabetic words: alpha_only

alpha_only = [t for t in lower_tokens if t.isalpha()]

print('List containing only Alphabetical tokens:',alpha_only)



# Remove all stop words: no_stops from the list now obtained

no_stops = [t for t in alpha_only if t not in stopWords]

print('Sentence without stop words: ',no_stops)



# Instantiate the WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()



# Lemmatize all tokens into a new list: lemmatized

lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

print("Lemmatized Tokens: ",lemmatized)



# Create the bag-of-words: bow

bow = Counter(lemmatized)

print('Bag Of Words: ',bow)



# Print the 3 most common tokens

print('Top 3 most common tokens: ',bow.most_common())

import nltk

from nltk.corpus import state_union

from nltk.tokenize import PunktSentenceTokenizer
"""

TRAINING DATA

"""

train_text = state_union.raw("2005-GWBush.txt") #State of the Union address from 2005

# print(train_text)

"""

TESTING DATA

"""

sample_text = state_union.raw("2006-GWBush.txt") #President George W. Bush's address from 2006

# print(sample_text)



# Tokenise the TRAINING text and train Punkt tokenizer

# custom_sent_tokenizer object is created of type PunktSentenceTokenizer

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)





# tokenise the TESTING text

tokenized = custom_sent_tokenizer.tokenize(sample_text)

# print(tokenized)





"""

SPEECH TAGGING FUNCTION

"""

def process_content():

    try:

        for i in tokenized[:5]:

            # tokenise on 'word entities' using nltk.word_tokenize

            words = nltk.word_tokenize(i)

            # tag each word with their respective pos value

            tagged = nltk.pos_tag(words)

            # Reference : https://www.nltk.org/book/ch05.html

            # prints the tuples ('word', 'pos')

            print(tagged)



    except Exception as e:

        print(str(e))





process_content()



import nltk

from nltk.corpus import state_union

from nltk.tokenize import PunktSentenceTokenizer



train_text = state_union.raw("2005-GWBush.txt")

sample_text = state_union.raw("2006-GWBush.txt")



custom_sent_tokenizer = PunktSentenceTokenizer(train_text)



tokenized = custom_sent_tokenizer.tokenize(sample_text)



def process_content():

    try:

        for i in tokenized:

            words = nltk.word_tokenize(i)

            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

            chunkParser = nltk.RegexpParser(chunkGram)

            chunked = chunkParser.parse(tagged)

            chunked.draw()     



    except Exception as e:

        print(str(e))



process_content()
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer

from nltk.corpus import gutenberg



# sample text

sample = gutenberg.raw("bible-kjv.txt")

print(sample)

tok = sent_tokenize(sample)



# for x in range(4):

#     print(tok[x])
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()



print(lemmatizer.lemmatize("cats"))

print(lemmatizer.lemmatize("cacti"))

print(lemmatizer.lemmatize("geese"))

print(lemmatizer.lemmatize("rocks"))

print(lemmatizer.lemmatize("python"))

print(lemmatizer.lemmatize("better", pos="a"))

print(lemmatizer.lemmatize("best", pos="a"))

print(lemmatizer.lemmatize("run"))

print(lemmatizer.lemmatize("run",'v'))
from textblob import TextBlob



analysis = TextBlob("TextBlob sure looks like it has some interesting features!")

# analysis

print(analysis.sentiment)
import nltk

import random

from nltk.corpus import movie_reviews



# movie_reviews [/usr/share/nltk_data/corpora/movie_reviews]

documents = [(list(movie_reviews.words(fileid)), category)

             for category in movie_reviews.categories()

             for fileid in movie_reviews.fileids(category)]



random.shuffle(documents)



# print(documents[1])  [positive / negative tags]



all_words = []

#cast words to lower-case

for w in movie_reviews.words():

    all_words.append(w.lower())

#convert into a  frequency distribution

all_words = nltk.FreqDist(all_words)

#top 15 most common words

print(all_words.most_common(15))

# print(all_words["stupid"])



import nltk

import random

from nltk.corpus import movie_reviews



documents = [(list(movie_reviews.words(fileid)), category)

             for category in movie_reviews.categories()

             for fileid in movie_reviews.fileids(category)]



random.shuffle(documents)



all_words = []



for w in movie_reviews.words():

    all_words.append(w.lower())



all_words = nltk.FreqDist(all_words)



word_features = list(all_words.keys())[:3000]

word_features
def find_features(document):

    words = set(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)



    return features
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

featuresets
# set that we'll train our classifier with

training_set = featuresets[:1900]



# set that we'll test against.

testing_set = featuresets[1900:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
import pickle
save_classifier = open("naivebayes.pickle","wb")

pickle.dump(classifier, save_classifier)

save_classifier.close()
classifier_f = open("naivebayes.pickle", "rb")

classifier = pickle.load(classifier_f)

print(classifier)

classifier_f.close()
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB,BernoulliNB

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC
MNB_classifier = SklearnClassifier(MultinomialNB())

MNB_classifier.train(training_set)

print("MultinomialNB accuracy percent:",(nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)



BNB_classifier = SklearnClassifier(BernoulliNB())

BNB_classifier.train(training_set)

print("BernoulliNB accuracy percent:",(nltk.classify.accuracy(BNB_classifier, testing_set)) * 100)



print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)



MNB_classifier = SklearnClassifier(MultinomialNB())

MNB_classifier.train(training_set)

print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



BernoulliNB_classifier = SklearnClassifier(BernoulliNB())

BernoulliNB_classifier.train(training_set)

print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)



LogisticRegression_classifier = SklearnClassifier(LogisticRegression())

LogisticRegression_classifier.train(training_set)

print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)



SGDClassifier_classifier = SklearnClassifier(SGDClassifier())

SGDClassifier_classifier.train(training_set)

print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)



SVC_classifier = SklearnClassifier(SVC())

SVC_classifier.train(training_set)

print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)



LinearSVC_classifier = SklearnClassifier(LinearSVC())

LinearSVC_classifier.train(training_set)

print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)



NuSVC_classifier = SklearnClassifier(NuSVC())

NuSVC_classifier.train(training_set)

print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

import nltk

import random

from nltk.corpus import movie_reviews

from nltk.classify.scikitlearn import SklearnClassifier

import pickle



from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC



from nltk.classify import ClassifierI

from statistics import mode





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



documents = [(list(movie_reviews.words(fileid)), category)

             for category in movie_reviews.categories()

             for fileid in movie_reviews.fileids(category)]



random.shuffle(documents)



all_words = []



for w in movie_reviews.words():

    all_words.append(w.lower())



all_words = nltk.FreqDist(all_words)



word_features = list(all_words.keys())[:3000]



def find_features(document):

    words = set(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)



    return features



#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))



featuresets = [(find_features(rev), category) for (rev, category) in documents]

        

training_set = featuresets[:1900]

testing_set =  featuresets[1900:]



#classifier = nltk.NaiveBayesClassifier.train(training_set)



classifier_f = open("naivebayes.pickle","rb")

classifier = pickle.load(classifier_f)

classifier_f.close()



print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)



MNB_classifier = SklearnClassifier(MultinomialNB())

MNB_classifier.train(training_set)

print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



BernoulliNB_classifier = SklearnClassifier(BernoulliNB())

BernoulliNB_classifier.train(training_set)

print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)



LogisticRegression_classifier = SklearnClassifier(LogisticRegression())

LogisticRegression_classifier.train(training_set)

print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)



SGDClassifier_classifier = SklearnClassifier(SGDClassifier())

SGDClassifier_classifier.train(training_set)

print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)



##SVC_classifier = SklearnClassifier(SVC())

##SVC_classifier.train(training_set)

##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)



LinearSVC_classifier = SklearnClassifier(LinearSVC())

LinearSVC_classifier.train(training_set)

print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)



NuSVC_classifier = SklearnClassifier(NuSVC())

NuSVC_classifier.train(training_set)

print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)





voted_classifier = VoteClassifier(classifier,

                                  NuSVC_classifier,

                                  LinearSVC_classifier,

                                  SGDClassifier_classifier,

                                  MNB_classifier,

                                  BernoulliNB_classifier,

                                  LogisticRegression_classifier)



print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)



print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)

print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)

print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)

print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)

print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)

print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)
# import os 

# print(os.listdir("../input"))
def isItUnicode(filename):

    with open(filename, 'rb') as f:

#         encodingInfo = chardet.detect(f.read())

        encodingInfo = chardet.detect(f.encode("utf-8", errors = "replace").read())

        print(encodingInfo['encoding'])



# isItUnicode("../input/negative.txt") 



# isItUnicode("../input/positive.txt") 





short_pos = open("../input/negative.txt","r").read()

# short_pos.decode("utf-8").raw
from subprocess import check_output

# print(check_output(["ls", "../input/negative.txt"]).decode("utf8"))

content = check_output(["ls", "../input/negative.txt"]).decode("utf8")

content



# open("../input/negative.txt", 'rb').decode("utf8")
with open("../input/negative.txt", encoding="utf-8") as f:

    # read in 5000 bytes from our text file

    lines = f.readlines(5000)