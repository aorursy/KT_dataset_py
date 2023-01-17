import numpy as np

import pandas as pd

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from bs4 import BeautifulSoup

import re

from nltk.stem.wordnet import WordNetLemmatizer
train_docs = [

    "He is a lazy boy. She is also lazy.",

    "Edwin is a lazy person."

]



test_docs = [

    "She is extremely lazy."

]
lemmatizer = WordNetLemmatizer() # Instantiate lemmatizer



def proc_text(messy): #input is a single string

    first = BeautifulSoup(messy, "lxml").get_text() #gets text without tags or markup, remove html

    second = re.sub("[^a-zA-Z]"," ",first) #obtain only letters

    third = second.lower() #obtains a list of words in lower case

    fourth = word_tokenize(third) # tokenize

    fifth = [lemmatizer.lemmatize(str(x)) for x in fourth] #lemmatizing

    stops = set(stopwords.words("english")) #faster to search through a set than a list

    final = [w for w in fifth if not w in stops] #remove stop words

    return final
train = [proc_text(text) for text in train_docs] # preprocess training set

test = [proc_text(text) for text in test_docs] # preprocess test set



print("Training Documents:\n{}\n\n".format(train))

print("Test Documents:\n{}\n\n".format(test))
# Construct Vocabulary-Index Dictionary



vocab_idx = dict()

num_vocab = len(vocab_idx)



for doc in train:

    for token in set(doc):

        if token not in vocab_idx.keys():

            vocab_idx[token] = num_vocab

            num_vocab += 1
print("A Peek at Vocabulary-Index Dictionary\n\n")

print(vocab_idx)
# Construct Matrix of Zeroes



train_matrix = np.zeros((len(train), num_vocab)) # Document Term Matrix for training set
# Update Counts of Terms in Each Document for Training Set



for idx, doc in enumerate(train):

    for token in doc:

        train_matrix[idx, vocab_idx[token]] += 1
print("A Peek at Train Matrix\n\n")

print(train_matrix)
# Construct a matrix of zeros



test_matrix = np.zeros((len(test), num_vocab))
# Update Counts of Terms in Each Document for Test Set



for idx, doc in enumerate(test):

    for token in doc:

        if token in vocab_idx.keys():

            test_matrix[idx, vocab_idx[token]] += 1
print("A Peek at Test Matrix\n\n")

print(test_matrix)