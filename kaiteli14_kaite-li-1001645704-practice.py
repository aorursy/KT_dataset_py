# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Importing all the needed libraries in one place.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

import seaborn as sns

sns.set(style='darkgrid')

import matplotlib.pyplot as plt

import string

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

from sklearn import feature_extraction, linear_model, model_selection, preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
print("Train dataset size: ", len(train)) # size = 7613

print("Test dataset size: ", len(test)) # size = 3263

print("----------------------------------------------------------------------------------")

print(train.head(5))

print("----------------------------------------------------------------------------------")

print(test.head(5))



positive = train['target'].value_counts()[0]

negative = train['target'].value_counts()[1]

total = train.shape[0]

positive_ratio = positive / total;

negative_ratio = negative / total;





print(positive)

print(negative)

print(total)



plt.figure(figsize=(10,8))

sns.countplot(train['target'])



plt.xlabel('Real or Fake', size=15, labelpad=15)

plt.ylabel('Tweet Count', size=15, labelpad=15)



plt.xticks((0, 1),[

  'Real Tweet ({0:.2f}%)'.format(positive_ratio),

  'Fake Tweet ({0:.2f}%)'.format(negative_ratio)      

])



plt.tick_params(axis='x', labelsize=13)

plt.tick_params(axis='y', labelsize=13)



plt.title('Real or Diaster Tweet Distribution', size=15, y=1.05)



plt.show()



joined_all_data = pd.concat([train, test], axis = 0, sort = False)

print("Unique locations size after joinning: ",  joined_all_data.location.nunique()) # size = 4521

print("Unique keywords size: ", joined_all_data.keyword.nunique()) # size = 221

print("% of real disaster vs total number: ", joined_all_data.target.sum() / len(train)) # ratio = 0.42960,ratio of the real disaster tweets

print("Missing value: ")

joined_all_data.isna().sum()
joined_all_data.location.fillna("N/A", inplace = True)

joined_all_data.keyword.fillna("N/A", inplace = True)
train["keyword"] = train["keyword"].str.lower()

train["text"] = train["text"].str.lower()

test["keyword"] = test["keyword"].str.lower()

test["text"] = test["text"].str.lower()
string_punctuation = string.punctuation

def punctuationRemoval(t):

    return t.translate(t.maketrans('','',string_punctuation))



train["text"] = train["text"].apply(lambda text: punctuationRemoval(text))

test["text"] = test["text"].apply(lambda text: punctuationRemoval(text))

import re

import unidecode



def miscellaneousRemoval(t):

    pattern_url = re.compile(r'https?://\S+\www\.\S+')

    pattern_html = re.compile(r'<.*?>')

    

    t = pattern_url.sub(r'', t)

    t = pattern_html.sub(r'', t)

    t = unidecode.unidecode(t)

    

    return t



train["text"] = train["text"].apply(lambda text: miscellaneousRemoval(text))

test["text"] = test["text"].apply(lambda text: miscellaneousRemoval(text))
vectorizer = feature_extraction.text.CountVectorizer()

# example = vectorizer.fit_transform(train["text"][0:5])

# print(example[0].todense().shape)

# print(example[0].todense())



train_v = vectorizer.fit_transform(train["text"])

test_v = vectorizer.transform(test["text"])
# I have tried both the linear classifier in "RidgeClasssifier()" and the Multinomial Naive Bya classifier from "sklearn"; it turns out that Naive Bay method

# is a little bit better for text classifier. So I finalized my submission with Naive Bay(only less than 2% more accuracy).



### Linear Classifier

# linear_classifier = linear_model.RidgeClassifier()

# score = model_selection.cross_val_score(linear_classifier, train_v, train["target"], cv=3, scoring="f1")

# score

# linear_classifier.fit(train_v, train["target"])



# submission["target"] = linear_classifier.predict(test_v)

# submission.head()

# submission.to_csv("submission.csv", index=False)



### Naive Classifier

from sklearn.naive_bayes import MultinomialNB as NB

naive_bay_classifier = NB()

score = model_selection.cross_val_score(naive_bay_classifier, train_v, train["target"], cv=3, scoring="f1")

score

naive_bay_classifier.fit(train_v, train["target"])

submission["target"] = naive_bay_classifier.predict(test_v)

submission.head()

submission.to_csv("submission.csv", index=False)