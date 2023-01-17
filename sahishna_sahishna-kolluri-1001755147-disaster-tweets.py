import numpy as np 

import pandas as pd 

import seaborn as sb

import matplotlib.pyplot as pyp

import re

import string

import os



from string import punctuation

from collections import defaultdict

from nltk import FreqDist

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import CountVectorizer as countVectorizer

from sklearn.model_selection import train_test_split

from sklearn import feature_extraction, linear_model, model_selection, preprocessing



train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
submission.head(10)
print('There are {} rows and {} columns in train'.format(train_df.shape[0],train_df.shape[1]))

print('There are {} rows and {} columns in test'.format(test_df.shape[0],test_df.shape[1]))
x = train_df.target.value_counts()

sb.barplot(x.index, x)

pyp.gca().set_ylabel('Samples')
fig, (ax1, ax2) = pyp.subplots(1, 2, figsize=(10,5))

word = train_df[train_df['target']==1]['text'].str.split().apply(lambda x: [len(i) for i in x])

sb.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='red')

ax1.set_title('Disaster tweets')

word = train_df[train_df['target']==0]['text'].str.split().apply(lambda x: [len(i) for i in x])

sb.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='green')

ax2.set_title('Non disaster tweets')

fig.suptitle('Average word length in each tweet')
pyp.figure(figsize=(9,6))

sb.countplot(y=train_df.keyword, order = train_df.keyword.value_counts().iloc[:20].index)

pyp.title('Most Used keywords')

pyp.show()
pyp.figure(figsize=(9,6))

sb.countplot(y=train_df.location, order = train_df.location.value_counts().iloc[:20].index)

pyp.title('Top 20 Locations')

pyp.show()
has_keyword = train_df['keyword'].isna().value_counts()

has_location = train_df['location'].isna().value_counts()



fig, axs = pyp.subplots(ncols=2, figsize = (12, 5))

sb.barplot(has_keyword.index, has_keyword, ax = axs[0])

sb.barplot(has_location.index, has_location, ax = axs[1])
def get_top_tweet_bigram(corpus, n=None):

    vec = countVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

pyp.figure(figsize=(10,5))

top_tweet_bigram = get_top_tweet_bigram(train_df['text'].tolist())[:10]

x,y = map(list, zip(*top_tweet_bigram))

sb.barplot(y,x)
data = pd.concat([train_df,test_df], axis = 0, sort=False)



print("Number of unique locations: ", data.location.nunique())





print("Missing values:")

data.isna().sum()

data.location.fillna("None", inplace=True)

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
data['text']=data['text'].apply(lambda x : remove_URL(x))

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)
data['text']=data['text'].apply(lambda x : remove_punct(x))

vectorizer = feature_extraction.text.CountVectorizer()

train_v = vectorizer.fit_transform(train_df["text"])

test_v = vectorizer.transform(test_df["text"])
linear_classifier = linear_model.RidgeClassifier()

score = model_selection.cross_val_score(linear_classifier, train_v, train_df["target"], cv=3, scoring="f1")

score

linear_classifier.fit(train_v, train_df["target"])
submission["target"] = linear_classifier.predict(test_v)

submission.head()
submission.to_csv('submission.csv', index=False)