import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import nltk

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from collections import Counter

from gensim.models import word2vec

from scipy.spatial.distance import *

from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial.distance import pdist

import string

from random import randint

from numpy import add

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
STOP_WORDS = nltk.corpus.stopwords.words()



def clean_sentence(line):

    st = ""

    line = re.sub(r'[^\w\s]','',line)

    for letter in line:

        if ord(letter) > 127:

            st += chr(randint(97,122))

        else:

            st += letter.lower()

    return st



def clean_dataframe(data):

    "drop nans, then apply 'clean_sentence' function to question1 and 2"

    data = data.dropna(how="any")

    

    for col in ['question1', 'question2']:

        data[col] = data[col].apply(clean_sentence)

    

    return data



train = clean_dataframe(train)

train.head(10)
STOP_WORDS = nltk.corpus.stopwords.words()



def clean_sentence(line):

    st = ""

    line = re.sub(r'[^\w\s]','',line)

    for letter in line:

        if ord(letter) > 127:

            st += chr(randint(97,122))

        else:

            st += letter.lower()

    return st



def clean_dataframe(data):

    "drop nans, then apply 'clean_sentence' function to question1 and 2"

    data = data.dropna(how="any")

    

    for col in ['question1', 'question2']:

        data[col] = data[col].apply(clean_sentence)

    

    return data



test = clean_dataframe(test)

test.head(10)
def build_corpus(data):

    corpus = []

    for col in ['question1', 'question2']:

        for sentence in data[col].iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

            

    return corpus



train_corpus = build_corpus(train)        

train_corpus[0:2]
def build_corpus(data):

    corpus = []

    for col in ['question1', 'question2']:

        for sentence in data[col].iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

            

    return corpus



test_corpus = build_corpus(test)        

test_corpus[0:2]
train_model = word2vec.Word2Vec(train_corpus, size=100, window=20, min_count=1, workers=4)
test_model = word2vec.Word2Vec(test_corpus, size=100, window=20, min_count=1, workers=4)
train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)



# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 

words = (" ".join(train_qs)).lower().split()

counts = Counter(words)

train_weights = {word: get_weight(count) for word, count in counts.items()}
test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist()).astype(str)



# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 

words = (" ".join(train_qs)).lower().split()

counts = Counter(words)

test_weights = {word: get_weight(count) for word, count in counts.items()}
def weight_vector(sentence):

    vector = []

    total = 0

    for word in sentence:

        if( len(word) == 0):

            pass

        m = model.wv[word]

        w = weights.get(word)

        vector.append(m)

        vector.append(w)

            

    try:

        for array in vector:

            if(array != None):

                total = add(total, array)

                

        vector = np.mean(vector)

        return vector

    except:

        return None

        

        

    
'''

labels = []

for i in train.is_duplicate:

    labels.append(i)

distance = []

for a,b in zip(train.question1, train.question2): 

    vec1 = weight_vector(a.split(' '))

    vec2 = weight_vector(b.split(' '))

    x = cosine(vec1,vec2)

       

    if np.isnan(x) == False:

        distance.append(x)

    else:

        distance.append(-1)



print(len(labels))

print(len(distance))

'''

'''

distance = []

for a,b in zip(test.question1, test.question2): 

    vec1 = weight_vector(a.split(' '))

    vec2 = weight_vector(b.split(' '))

    x = cosine(vec1,vec2)

       

    if np.isnan(x) == False:

        distance.append(x)

    else:

        distance.append(-1)



print(len(distance))

'''

train_labels = []

for i in train.is_duplicate:

    train_labels.append(i)

train_distance = []

for a,b in zip(train.question1, train.question2): 

    vec1 = weight_vector(a.split(' '))

    vec2 = weight_vector(b.split(' '))

    x = sqeuclidean(vec1,vec2)

       

    if np.isnan(x) == False:

        train_distance.append(x)

    else:

        train_distance.append(-1)



print(len(train_labels))

print(len(train_distance))
test_distance = []

for a,b in zip(test.question1, test.question2): 

    vec1 = weight_vector(a.split(' '))

    vec2 = weight_vector(b.split(' '))

    x = sqeuclidean(vec1,vec2)

       

    if np.isnan(x) == False:

        test_distance.append(x)

    else:

        test_distance.append(-1)



print(len(test_distance))
train_data = []

for i in range(len(train_labels)):

    if train_distance[i] > 0:

        train_data.append([train_labels[i], train_distance[i]])

len(train_data)
test_data = []

for i in test_distance:

    if i > 0:

        test_data.append(i)

len(test_data)
forest = RandomForestClassifier(max_depth = 25, min_samples_split=10, min_samples_leaf=10, n_estimators = 1000, random_state = 1) 

forest_model = forest.fit(labelled_points, test_data)
print(forest_model.score(features_forest, target))


