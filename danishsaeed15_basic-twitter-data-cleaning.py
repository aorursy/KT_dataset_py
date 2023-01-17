# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import re

import nltk

import time

import numpy as np

import pandas as pd

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

import gensim.parsing.preprocessing as gpp



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gensim

print("List of stopwords:\n")

print(gensim.parsing.preprocessing.STOPWORDS)
# Custom function for counting words in a list of text data

def count_words(mylist):

    w = 0

    for items in mylist.iteritems():

        w += len(items[1].split())

    return w



# Map POS tag to first character lemmatize() accepts

def get_wordnet_pos(word):

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
tweets = pd.read_csv("/kaggle/input/tweets-with-keyword-lockdown-in-apriljuly-2020/tweets_lockdown.csv",

                     index_col=0)



words_start = count_words(tweets["Text"])



print("Number of rows: {}".format(tweets.shape[0]))

print("Number of words: {}".format(words_start))

tweets.head(5)
# Removing links and ampersand attached text from the tweets

text_list = [re.sub(r"(?:\@|\&|http)\S+", "", item) for item in tweets["Text"]]



# Removing numeric data from tweets

text_list = [gpp.strip_numeric(item) for item in text_list]



# Removing non alphanumeric characters from tweets

text_list = [gpp.strip_non_alphanum(item) for item in text_list]



# Removing punctuation symbols from tweets

text_list = [gpp.strip_punctuation(item) for item in text_list]



# Stripping short words with length less than minsize

text_list = [gpp.strip_short(item, minsize=2) for item in text_list]



# Convert everything to lowercase

text_list = [item.lower() for item in text_list]



# Removing stopwords from the text based on gensim stopwords list

text_list = [gpp.remove_stopwords(item) for item in text_list]



# Alternate unverified method to remove everything except text and numbers

# text_list["text"] = [re.sub(r"[^a-zA-Z0-9]+", ' ', item) for item in text_list["text"]]



tweets_text = pd.DataFrame(text_list, columns=["Text"])

tweets_text.dropna(inplace=True)

print("Number of rows: {}".format(tweets_text.shape[0]))

words_cleaning = count_words(tweets_text["Text"])

print("Number of words: {}".format(words_cleaning))

print("Number of words removed in text cleaning: {}".format(words_start-words_cleaning))



tweets_text.head(5)
# custom_words are words to not remove while checking in english dictionary

# For example, the word "lockdown" is not in the dictionary

custom_words = ["lockdown"]



print("\nRunning lemmatization ...\n")



start = time.time()

lemma = nltk.wordnet.WordNetLemmatizer()

words = set(nltk.corpus.words.words())



# Adding custom words in the dictinary so they are not removed

for i in custom_words:

    words.add(i)



tweets_ll = []

for item in tweets_text["Text"]:

    word_list = item.split()



    # Lemmatization of each word 

    word_list = [lemma.lemmatize(x, get_wordnet_pos(x)) for x in word_list]



    # Checking for words in nltk english dictionary

    word_list = [x for x in word_list if x in words]

    tweets_ll.append(" ".join(word_list))

    word_list = None



end = time.time()

tweets_final = pd.DataFrame(tweets_ll, columns=["Text"])

tweets_final.dropna(inplace=True)

print("Total time taken in lemmatization: {:.2f} seconds".format(end-start))
print("Number of rows: {}".format(tweets_final.shape[0]))

words_lemmatized = count_words(tweets_final["Text"])

print("Number of words: {}".format(words_lemmatized))

print("Number of words removed in lemmatization: {}".format(words_cleaning-words_lemmatized))

tweets_final.head(5)
tweets_final.to_csv("tweets_cleaned_text.csv")

tweets_final.head(5)