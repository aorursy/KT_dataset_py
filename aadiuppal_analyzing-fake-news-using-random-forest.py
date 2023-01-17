# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



fake_news = pd.read_csv("../input/fake.csv")

fake_news.head(10)



# Any results you write to the current directory are saved as output.
fake_news.type.value_counts().plot(kind='bar')
from sklearn.model_selection import train_test_split

train, test = train_test_split(fake_news, test_size = 0.2)

#print(len(train),len(test))



train_one = train[train["language"]=="english"]

test_one = test[test["language"]=="english"]

#train_one.columns.values

#print(len(test_one),len(train_one))

#import nltk

from nltk.corpus import stopwords 

train.columns.values

#Text_col = train["text"]

#Author_col = train["author"]

#Site_col = train["site_url"]

#Title_col = train["title"]

#Thread_col = train["thread_title"]

train.head()

import re

def refineWords(s):

    letters_only = re.sub("[^a-zA-Z]", " ", s) 

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    #print( " ".join( meaningful_words ))

    return( " ".join( meaningful_words ))

train_one["text"].fillna(" ",inplace=True)    

train_one["text"] = train_one["text"].apply(refineWords)

train_one["author"].fillna(" ",inplace=True)    

train_one["author"] = train_one["author"].apply(refineWords)

train_one["site_url"].fillna(" ",inplace=True)    

train_one["site_url"] = train_one["site_url"].apply(refineWords)

train_one["title"].fillna(" ",inplace=True)    

train_one["title"] = train_one["title"].apply(refineWords)

train_one["thread_title"].fillna(" ",inplace=True)    

train_one["thread_title"] = train_one["thread_title"].apply(refineWords)

train_two = train_one.copy()

train_one.head()
train_one = train_two.copy()

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 5000) 

#print(train_one["title"].head())

#temp  = (vectorizer.fit_transform(train_one["text"]))

#train_one["text"] = temp.to_array()

train_one["text"] = vectorizer.fit_transform(train_one["text"]).toarray()

train_one["author"] = vectorizer.fit_transform(train_one["author"]).toarray()

train_one["site_url"] = vectorizer.fit_transform(train_one["site_url"]).toarray()

train_one["title"] = vectorizer.fit_transform(train_one["title"]).toarray()

train_one["thread_title"] = vectorizer.fit_transform(train_one["thread_title"]).toarray()

train_one.head()
#print((train_one["text"][11543]).shape)

#print(type(temp))

print(train_one.describe())

dist = np.sum(train_one, axis=0)
train_one["domain_rank"].fillna(train_one.domain_rank.median(axis=0),inplace=True)

test_one["domain_rank"].fillna(test_one.domain_rank.median(axis=0),inplace=True)  
train_one["isSpam"] = np.sign(train_one["spam_score"]-0.5)

#print(train_one["isSpam"])

from sklearn.ensemble import RandomForestClassifier

#forest = RandomForestClassifier(n_estimators = 100)

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

features_forest = train_one[["text", "author", "site_url", "title", "thread_title","domain_rank"]].values

my_forest = forest.fit(features_forest, train_one["isSpam"])
target = train_one["isSpam"].values

print(my_forest.score(features_forest, target))
test_one["text"].fillna(" ",inplace=True)    

test_one["text"] = test_one["text"].apply(refineWords)

test_one["author"].fillna(" ",inplace=True)    

test_one["author"] = test_one["author"].apply(refineWords)

test_one["site_url"].fillna(" ",inplace=True)    

test_one["site_url"] = test_one["site_url"].apply(refineWords)

test_one["title"].fillna(" ",inplace=True)    

test_one["title"] = test_one["title"].apply(refineWords)

test_one["thread_title"].fillna(" ",inplace=True)    

test_one["thread_title"] = test_one["thread_title"].apply(refineWords)

test_two = test_one.copy()







test_one["text"] = vectorizer.fit_transform(test_one["text"]).toarray()

test_one["author"] = vectorizer.fit_transform(test_one["author"]).toarray()

test_one["site_url"] = vectorizer.fit_transform(test_one["site_url"]).toarray()

test_one["title"] = vectorizer.fit_transform(test_one["title"]).toarray()

test_one["thread_title"] = vectorizer.fit_transform(test_one["thread_title"]).toarray()

test_one["isSpam"] = np.sign(test_one["spam_score"]-0.5)
test_features = test_one[["text", "author", "site_url", "title", "thread_title","domain_rank"]].values

my_prediction = my_forest.predict(test_features)

print(len(my_prediction),len(test_one["isSpam"]))
count = 0

pred = my_prediction.tolist()

test_spam = test_one["isSpam"].tolist()

for i in range(len(pred)):

    if pred[i] == test_spam[i]:

        count += 1

print(count,float(count)/len(my_prediction))

#print(my_prediction)

#print(test_spam)