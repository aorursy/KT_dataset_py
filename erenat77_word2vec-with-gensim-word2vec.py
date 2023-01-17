# importing dataframes and array operations

import pandas as pd

import numpy as np

# BeautifulSoup is used to remove html tags from the text

from bs4 import BeautifulSoup 

import re # for regular expression



# Stopwords can be useful to undersand the semantics of the sentence.

# Therefore stopwords are not removed while creating the word2vec model.

# But they will be removed  while averaging feature vectors.

from nltk.corpus import stopwords
# reading .tsv file

train = pd.read_csv("../input/word2vec/unlabeledTrainData.tsv", header=0,\

                    delimiter="\t", quoting=3)
#visualize the context

train.head()
# checking for Nan or empty strings

train.isnull().sum()
# This function converts a text to a sequence of words.

def review_wordlist(review, remove_stopwords=False):

    # 1. Removing html tags

    review_text = BeautifulSoup(review).get_text()

    # 2. Removing non-letter.

    review_text = re.sub("[^a-zA-Z]"," ",review_text)

    # 3. Converting to lower case and splitting

    words = review_text.lower().split()

    # 4. Optionally remove stopwords

    if remove_stopwords:

        stops = set(stopwords.words("english"))     

        words = [w for w in words if not w in stops]

    #5. lemma

    

    return(words)
# word2vec expects a list of lists.

# Using punkt tokenizer for better splitting of a paragraph into sentences.



import nltk.data

#nltk.download('popular')



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# This function splits a review into sentences

def review_sentences(review, tokenizer, remove_stopwords=False):

    # 1. Using nltk tokenizer

    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []

    # 2. Loop for each sentence

    for raw_sentence in raw_sentences:

        if len(raw_sentence)>0:

            sentences.append(review_wordlist(raw_sentence,\

                                            remove_stopwords))



    # This returns the list of lists

    return sentences
sentences = []

print("Parsing sentences from training set")

for review in train["review"]:

    sentences += review_sentences(review, tokenizer)
# Creating the model and setting values for the various parameters

num_features = 300  # Word vector dimensionality

min_word_count = 40 # Minimum word count

num_workers = 4     # Number of parallel threads

context = 10        # Context window size

downsampling = 1e-3 # (0.001) Downsample setting for frequent words



# Initializing the train model

from gensim.models import word2vec

print("Training model....")

model = word2vec.Word2Vec(sentences,\

                          workers=num_workers,\

                          size=num_features,\

                          min_count=min_word_count,\

                          window=context,

                          sample=downsampling)



# To make the model memory efficient

model.init_sims(replace=True)



# Saving the model for later use. Can be loaded using Word2Vec.load()

print("Saving the model")

model_name = "300features_40minwords_10context"

model.save(model_name)

# Few tests: This will print the odd word among them 

model.wv.doesnt_match("man woman king queen princess dog".split())
model.wv.doesnt_match("europe africa USA turkey".split())
model.wv.most_similar("best")
model.wv.most_similar("boring")
model.wv.most_similar_cosmul(positive=['man', 'woman'], negative=['princess'])
model.wv.syn0.shape