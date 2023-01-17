# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import string

from sklearn.feature_extraction.text import CountVectorizer



sns.set(style="darkgrid")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
# remove nan value

train = train[~train["text"].isna()]
train.head()
test.head()
sns.countplot(data=train, x="sentiment", order=["positive", "neutral", "negative"])
sns.countplot(data=test, x="sentiment", order=["positive", "neutral", "negative"])
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
train["word_count"] = train["text"].apply(lambda x: len(x.split()))

test["word_count"] = test["text"].apply(lambda x: len(x.split()))

train["selected_word_count"] = train["selected_text"].apply(lambda x: len(x.split()))

train["jaccard"] = train.apply(lambda x: jaccard(x["text"], x["selected_text"]), axis=1)
sns.distplot(train["word_count"], label="Train")

sns.distplot(test["word_count"], label="Test")

plt.legend()

plt.show()
sns.distplot(train["selected_word_count"])

plt.show()
sns.distplot(train["jaccard"])

plt.show()
train.groupby("sentiment")["word_count"].agg(["mean", "std", "min","median", "max"])
train.groupby("sentiment")["selected_word_count"].agg(["mean", "std", "min", "median", "max"])
train.groupby("sentiment")["jaccard"].agg(["mean", "std", "min", "median", "max"])
sns.boxplot(data=train, x="sentiment", y="word_count")

plt.show()
sns.boxplot(data=train, x="sentiment", y="selected_word_count")

plt.show()
sns.boxplot(data=train, x="sentiment", y="jaccard")

plt.show()
vec = CountVectorizer().fit(train["text"].apply(lambda x: x.lower()))

train_vocab = vec.vocabulary_.keys()

print(f"train vocab size: {len(train_vocab)}")
vec = CountVectorizer().fit(test["text"].apply(lambda x: x.lower()))

test_vocab = vec.vocabulary_.keys()

print(f"test vocab size: {len(test_vocab)}")
def get_test_only_vocab(train, test, col):

    vec = CountVectorizer().fit(train[col].apply(lambda x: x.lower()))

    train_vocab = vec.vocabulary_.keys()

    vec = CountVectorizer().fit(test[col].apply(lambda x: x.lower()))

    test_vocab = vec.vocabulary_.keys()

    test_only = set(test_vocab) - set(train_vocab)

    print(f"test only vocab size: {len(test_only)}")

    print(test_only)
get_test_only_vocab(train, test, "text")
get_test_only_vocab(train[train["sentiment"]=="positive"], test[test["sentiment"]=="positive"], "text")
get_test_only_vocab(train[train["sentiment"]=="neutral"], test[test["sentiment"]=="neutral"], "text")
get_test_only_vocab(train[train["sentiment"]=="negative"], test[test["sentiment"]=="negative"], "text")