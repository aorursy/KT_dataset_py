# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from bs4 import BeautifulSoup as bs

import re

from nltk.corpus import stopwords

import logging

from gensim.models import word2vec
# Read data from files 

train = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3 )

test = pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t", quoting=3 )

unlabeled_train = pd.read_csv("../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3 )
# Verify the number of reviews that were read (100,000 in total)

print(f"Read {train['review'].size} labeled train reviews, {test['review'].size} labeled test reviews and {unlabeled_train['review'].size} unlabeled reviews")
def review_to_wordlist( review, remove_stopwords=False ):

    '''Function to convert a document to a sequence of words,

    optionally removing stop words.  Returns a list of words.

    '''

    # 1. Remove HTML

    review_text = bs(review).get_text()

      

    # 2. Remove non-letters

    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    

    # 3. Convert words to lower case and split them

    words = review_text.lower().split()

    

    # 4. Optionally remove stop words (false by default)

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]

    

    # 5. Return a list of words

    return(words)
# Download the punkt tokenizer for sentence splitting

import nltk.data

#nltk.download()
# Load the punkt tokenizer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Define a function to split a review into parsed sentences

def review_to_sentences( review, tokenizer, remove_stopwords=False ):

    '''Function to split a review into parsed sentences. Returns a 

    list of sentences, where each sentence is a list of words

    '''

    # 1. Use the NLTK tokenizer to split the paragraph into sentences

    raw_sentences = tokenizer.tokenize(review.strip())

    

    # 2. Loop over each sentence

    sentences = []

    for raw_sentence in raw_sentences:

        # If a sentence is empty, skip it

        if len(raw_sentence) > 0:

            # Otherwise, call review_to_wordlist to get a list of words

            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

    

    # Return the list of sentences (each sentence is a list of words,

    # so this returns a list of lists

    return sentences
sentences = []  # Initialize an empty list of sentences



print("Parsing sentences from training set")

for review in train["review"]:

    sentences += review_to_sentences(review, tokenizer)



print("Parsing sentences from unlabeled set")

for review in unlabeled_train["review"]:

    sentences += review_to_sentences(review, tokenizer)
print(len(sentences))
print(sentences[0])
print(sentences[1])
# Import the built-in logging module and configure it so that Word2Vec 

# creates nice output messages

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Set values for various parameters

num_features = 300    # Word vector dimensionality                      

min_word_count = 40   # Minimum word count                        

num_workers = 4       # Number of threads to run in parallel

context = 10          # Context window size                                                                                    

downsampling = 1e-3   # Downsample setting for frequent words
# Initialize and train the model (this will take some time)

print("Training model...")

model = word2vec.Word2Vec(sentences,

                          workers=num_workers,

                          size=num_features,

                          min_count = min_word_count,

                          window = context,

                          sample = downsampling)

print("Finished training model.")
# If you don't plan to train the model any further, calling 

# init_sims will make the model much more memory-efficient.

model.init_sims(replace=True)
# It can be helpful to create a meaningful model name and 

# save the model for later use. You can load it later using Word2Vec.load()

model_name = "300features_40minwords_10context"

model.save(model_name)
model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("france england germany berlin".split())
model.doesnt_match("paris berlin london austria".split())
model.most_similar("man")
model.most_similar("queen")
model.most_similar("awful")