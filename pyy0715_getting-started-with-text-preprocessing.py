# First, look at everything.

from subprocess import check_output

print(check_output(["ls", "../input/customer-support-on-twitter"]).decode("utf8"))
import numpy as np

import pandas as pd

import re

import nltk

import spacy

import string

pd.options.mode.chained_assignment = None



full_df = pd.read_csv("../input/customer-support-on-twitter/twcs/twcs.csv", nrows=5000)

df = full_df[["text"]]

df["text"] = df["text"].astype(str)

full_df.head()
df["text_lower"] = df["text"].str.lower()

df.head()
# drop the new column created in last cell

df.drop(["text_lower"], axis=1, inplace=True)



PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):

    """custom function to remove the punctuation"""

    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



df["text_wo_punct"] = df["text"].apply(lambda text: remove_punctuation(text))

df.head()
from nltk.corpus import stopwords

", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])



df["text_wo_stop"] = df["text_wo_punct"].apply(lambda text: remove_stopwords(text))

df.head()
from collections import Counter

cnt = Counter()

for text in df["text_wo_stop"].values:

    for word in text.split():

        cnt[word] += 1

        

cnt.most_common(10)
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])

def remove_freqwords(text):

    """custom function to remove the frequent words"""

    return " ".join([word for word in str(text).split() if word not in FREQWORDS])



df["text_wo_stopfreq"] = df["text_wo_stop"].apply(lambda text: remove_freqwords(text))

df.head()
n_rare_words = 10

RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

print(RAREWORDS)
# Drop the two columns which are no more needed 

df.drop(["text_wo_punct", "text_wo_stop"], axis=1, inplace=True)



n_rare_words = 10

RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

def remove_rarewords(text):

    """custom function to remove the rare words"""

    return " ".join([word for word in str(text).split() if word not in RAREWORDS])



df["text_wo_stopfreqrare"] = df["text_wo_stopfreq"].apply(lambda text: remove_rarewords(text))

df.head()
from nltk.stem.porter import PorterStemmer



# Drop the two columns 

df.drop(["text_wo_stopfreq", "text_wo_stopfreqrare"], axis=1, inplace=True) 



stemmer = PorterStemmer()

def stem_words(text):

    return " ".join([stemmer.stem(word) for word in text.split()])



df["text_stemmed"] = df["text"].apply(lambda text: stem_words(text))

df.head()
from nltk.stem.snowball import SnowballStemmer

SnowballStemmer.languages
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])



df["text_lemmatized"] = df["text"].apply(lambda text: lemmatize_words(text))

df.head()
lemmatizer.lemmatize("running")
lemmatizer.lemmatize("running", "v") # v for verb
print("Word is : stripes")

print("Lemma result for verb : ",lemmatizer.lemmatize("stripes", 'v'))

print("Lemma result for noun : ",lemmatizer.lemmatize("stripes", 'n'))
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

print(wordnet_map.get(wordnet.NOUN))
from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatize_words(text):

    pos_tagged_text = nltk.pos_tag(text.split())

    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])



df["text_lemmatized"] = df["text"].apply(lambda text: lemmatize_words(text))

df.head()