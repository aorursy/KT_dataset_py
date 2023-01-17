# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import emoji

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from pyfasttext import FastText

import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix



import numpy as np

import pandas as pd



import warnings

import re

import itertools

import emoji

from io import StringIO

import csv



warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/train_2kmZucJ.csv")

print(train_df.shape)

test_df = pd.read_csv("/kaggle/input/test_oJQbWVk.csv")

print(test_df.shape)
train_df.head(10)
train_df['tweet'].iloc[0]
test_df.head(2)
# train_df["tweet"] = train_df["tweet"].apply(lambda x:' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", x).split()))

train_df["tweet"] = train_df["tweet"].apply(lambda x:' '.join(re.sub("@|#", " ", x).split()))



train_df["tweet"] = train_df["tweet"].apply(lambda x:' '.join(re.sub("(\w+:\/\/\S+)", " ", x).split()))

train_df["tweet"] = train_df["tweet"].apply(lambda x:' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", x).split()))

train_df["tweet"] = train_df["tweet"].apply(lambda x:x.lower())

train_df["tweet"] = train_df["tweet"].apply(lambda x:x.replace("’","'"))

train_df["tweet"] = train_df["tweet"].apply(lambda x:''.join(''.join(s)[:2] for _, s in itertools.groupby(x)))

train_df["tweet"] = train_df["tweet"].apply(lambda x:emoji.demojize(x))

train_df["tweet"] = train_df["tweet"].apply(lambda x:x.replace(":"," "))

train_df["tweet"] = train_df["tweet"].apply(lambda x:' '.join(x.split()))



test_df["tweet"] = test_df["tweet"].apply(lambda x:' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", x).split()))

test_df["tweet"] = test_df["tweet"].apply(lambda x:' '.join(re.sub("(\w+:\/\/\S+)", " ", x).split()))

test_df["tweet"] = test_df["tweet"].apply(lambda x:' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", x).split()))

test_df["tweet"] = test_df["tweet"].apply(lambda x:x.lower())

test_df["tweet"] = test_df["tweet"].apply(lambda x:x.replace("’","'"))

test_df["tweet"] = test_df["tweet"].apply(lambda x:''.join(''.join(s)[:2] for _, s in itertools.groupby(x)))

test_df["tweet"] = test_df["tweet"].apply(lambda x:emoji.demojize(x))

test_df["tweet"] = test_df["tweet"].apply(lambda x:x.replace(":"," "))

test_df["tweet"] = test_df["tweet"].apply(lambda x:' '.join(x.split()))
train_df.head(3)
import nltk

from nltk.corpus import stopwords

def tokenize_text(text):

    tokens = []

    for sent in nltk.sent_tokenize(text):

        for word in nltk.word_tokenize(sent):

            if len(word) < 2:

                continue

            tokens.append(word.lower())

    return tokens
tokenize_text("My name is bla bla")
import nltk

import gensim

from nltk.corpus import abc



model= gensim.models.Word2Vec(abc.sents())

X= list(model.wv.vocab)



data=model.most_similar('bottle')

print(data)
combi=train_df.append(test_df,ignore_index=True)

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer().fit(combi['tweet'])

print(len(bow_transformer.vocabulary_))
messages_bow = bow_transformer.transform(combi['tweet'])

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

"""

train_bow = messages_bow[:7920,:]

test_bow = messages_bow[7920:,:]



xtrain, xtest, ytrain, ytest = train_test_split(train_bow, train['label'],random_state=42,test_size=0.3)

"""

train_df['vector'] = bow_transformer.transform(train_df['tweet'])

bow_train = bow_transformer.transform(train_df['tweet'])

test_df['vector'] = bow_transformer.transform(test_df['tweet'])

bow_test = bow_transformer.transform(test_df['tweet'])

# xtrain,xtest,ytrain,ytest=train_test_split(train_bow,train_df['label'],random_state=42,test_size=0.3)
train_df['vector'].iloc[0]
xtrain,xtest,ytrain,ytest = train_test_split(bow_train,train_df['label'],random_state=42,test_size=0.3)
print(ytrain.shape,ytest.shape)
model=LogisticRegression()

model.fit(xtrain,ytrain)

prediction=model.predict_proba(xtest)

# train_df['label']
len(prediction)
prediction
prediction_int=prediction[:,1]>=0.25

prediction_int=prediction_int.astype(np.int)

f1_score(ytest,prediction_int)