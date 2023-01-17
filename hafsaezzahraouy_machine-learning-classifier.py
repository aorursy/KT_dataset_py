import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/train.csv",index_col=  0)
df.columns
df.size
# To change the names of DataFrame's Columns

df.columns = ["label","message"]
df.head()
# Change all the messages to string format

df['message'] = df['message'].apply(lambda x : str(x))
df["message"]
import nltk
df["length of message"] = df["message"].apply(len)
df.head()
# To plot a countplot to view how many SPAM messages are there and how many HAM

sns.countplot("label", data = df)
# To visualize the length of the messages based on the label

asdf = sns.FacetGrid(data = df, col = 'label')

asdf.map(sns.distplot, 'length of message', kde = False, hist_kws = dict(edgecolor = "k"))
from nltk.corpus import stopwords

import string
# These are the most common word which we have to remove from text messages

stopwords.words("french")
# We need to remove punctuations too

string.punctuation
# function for preprocessing

def all_words(msg):

    no_punctuation = [char for char in msg if char not in string.punctuation]

    no_punctuation = "".join(no_punctuation)

    word = [word for word in no_punctuation.split() if word.lower() not in stopwords.words("english")]

    return word
word=all_words(df["message"])
from collections import Counter
a=Counter(word)
len(a)
a.transform(df["message"])
print(bag_of_words_transformer)
len(bag_of_words_transformer)
# This will create the sparse matrix of all the messages based on the frequecy of words in that message

message_bow = bag_of_words_transformer.transform(df['message'])
# This is the shape of sparse matrix

# 310 is no. of message

# 1406 is the no. of words after preprocessing

message_bow.shape
tfid_transformer = TfidfTransformer().fit(message_bow)
message_tfid = tfid_transformer.transform(message_bow)
# Here Naive bayes has been used for traning

from sklearn.naive_bayes import MultinomialNB

spam_detection_model = MultinomialNB().fit(message_tfid,df['label'])
test = pd.read_csv("../input/test.csv", index_col = 0)
test.head()
test["'text'"] = test["'text'"].apply(lambda x: str(x))
test_message_bow = bag_of_words_transformer.transform(test["'text'"])
test_message_tfid = tfid_transformer.transform(test_message_bow)
# Prediction

test["'label'"] = spam_detection_model.predict(test_message_tfid)
sns.countplot(test["'label'"])