import os

import pandas as pd

import numpy as np



import nltk

from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.stem import SnowballStemmer

from nltk.stem import WordNetLemmatizer

import re

from string import punctuation
train_orig = pd.read_csv("../input/quora-question-pairs/train.csv")

test_orig = pd.read_csv("../input/quora-question-pairs/test.csv")

train_orig.head()
print(train_orig.isnull().sum())

print(test_orig.isnull().sum())
train_orig = train_orig.fillna(" ")

test_orig = test_orig.fillna(" ")
def common_words_transformation_remove_punctuation(text):

    

    text = text.lower()

    

    text = re.sub(r"what's", "what is", text)

    text = re.sub(r"who's", "who is", text)

    text = re.sub(r"where's", "where is", text)

    text = re.sub(r"when's", "when is", text)

    text = re.sub(r"how's", "how is", text)

    text = re.sub(r"it's", "it is", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"that's", "that is", text)

    text = re.sub(r"there's", "there is", text)



    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"\'s", " ", text)  # 除了上面的特殊情况外，“\'s”只能表示所有格，应替换成“ ”

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "can not ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r" m ", " am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"60k", " 60000 ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e-mail", "email", text)

    text = re.sub(r"\s{2,}", " ", text)

    text = re.sub(r"quikly", "quickly", text)

    text = re.sub(r" usa ", " america ", text)

    text = re.sub(r" u s ", " america ", text)

    text = re.sub(r" uk ", " england ", text)

    text = re.sub(r"imrovement", "improvement", text)

    text = re.sub(r"intially", "initially", text)

    text = re.sub(r" dms ", "direct messages ", text)  

    text = re.sub(r"demonitization", "demonetization", text) 

    text = re.sub(r"actived", "active", text)

    text = re.sub(r"kms", " kilometers ", text)

    text = re.sub(r" cs ", " computer science ", text)

    text = re.sub(r" ds ", " data science ", text)

    text = re.sub(r" ee ", " electronic engineering ", text)

    text = re.sub(r" upvotes ", " up votes ", text)

    text = re.sub(r" iphone ", " phone ", text)

    text = re.sub(r"\0rs ", " rs ", text) 

    text = re.sub(r"calender", "calendar", text)

    text = re.sub(r"ios", "operating system", text)

    text = re.sub(r"programing", "programming", text)

    text = re.sub(r"bestfriend", "best friend", text)

    text = re.sub(r"III", "3", text) 

    text = re.sub(r"the us", "america", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ", text)

    text = re.sub(r"\+", " ", text)

    text = re.sub(r"\-", " ", text)

    text = re.sub(r"\=", " ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " ", text)

    text = re.sub(r"\0s", "0", text)

    

    text = "".join([c for c in text if c not in punctuation])

        

    return text



train_orig["question1"] = train_orig["question1"].apply(common_words_transformation_remove_punctuation)

train_orig["question2"] = train_orig["question2"].apply(common_words_transformation_remove_punctuation)

test_orig["question1"] = test_orig["question1"].apply(common_words_transformation_remove_punctuation)

test_orig["question2"] = test_orig["question2"].apply(common_words_transformation_remove_punctuation)

train_orig.to_csv("train_orig.csv", index = False)

test_orig.to_csv("test_orig.csv", index = False)

train_orig.head()
stopwords.words("english")
def remove_stopwords(text):

    stops = set(stopwords.words("english"))

    text = word_tokenize(text)

    text = [w for w in text if not w in stops]

    text = " ".join(text)

    return text



train_stop, test_stop = train_orig.copy(deep = True), test_orig.copy(deep = True)

train_stop["question1"] = train_stop["question1"].apply(remove_stopwords)

train_stop["question2"] = train_stop["question2"].apply(remove_stopwords)

test_stop["question1"] = test_stop["question1"].apply(remove_stopwords)

test_stop["question2"] = test_stop["question2"].apply(remove_stopwords)

train_stop.to_csv("train_stop.csv", index = False)

test_stop.to_csv("test_stop.csv", index = False)

train_stop.head()
def stem_words(text):

    text = word_tokenize(text)

    stemmer = SnowballStemmer("english")

    stemmed_words = [stemmer.stem(word) for word in text]

    text = " ".join(stemmed_words)

    return text



train_stem, test_stem = train_stop.copy(deep = True), test_stop.copy(deep = True)

train_stem["question1"] = train_stem["question1"].apply(stem_words)

train_stem["question2"] = train_stem["question2"].apply(stem_words)

test_stem["question1"] = test_stem["question1"].apply(stem_words)

test_stem["question2"] = test_stem["question2"].apply(stem_words)

train_stem.to_csv("train_stem.csv", index = False)

test_stem.to_csv("test_stem.csv", index = False)

train_stem.head()
def lemmatize_words(text):

    text = word_tokenize(text)

    wordnet_lemmatizer = WordNetLemmatizer()

    lammatized_words = [wordnet_lemmatizer.lemmatize(word) for word in text]

    text = " ".join(lammatized_words)

    return text



train_lem, test_lem = train_stop.copy(deep = True), test_stop.copy(deep = True)

train_lem["question1"] = train_lem["question1"].apply(lemmatize_words)

train_lem["question2"] = train_lem["question2"].apply(lemmatize_words)

test_lem["question1"] = test_lem["question1"].apply(lemmatize_words)

test_lem["question2"] = test_lem["question2"].apply(lemmatize_words)

train_lem.to_csv("train_lem.csv", index = False)

test_lem.to_csv("test_lem.csv", index = False)

train_lem.head()