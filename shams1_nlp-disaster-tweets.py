import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

import re

plt.style.use('ggplot')
train_path = "../input/nlp-getting-started/train.csv"

test_path = "../input/nlp-getting-started/test.csv"

sample_submission_path = "../input/nlp-getting-started/sample_submission.csv"
df_train = pd.read_csv(train_path)

df_test = pd.read_csv(test_path)

submission = pd.read_csv(sample_submission_path)
df_train.head()
df_train.info()
print(df_train.info())

df_test.info()
df_train = df_train[['text','target']]

df_test = df_test[['text']]
y = np.array(df_train.target.value_counts())

sns.barplot(x = [0,1],y = y,palette='gnuplot2_r')

difference = y[0]-y[1]

print("Difference between target 0 and 1: ",y[0]-y[1])
df_train.text.describe()
df_train['Char_length'] = df_train['text'].apply(len)
f, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

f.suptitle("Histogram of char length of text",fontsize=20)

sns.distplot(df_train[df_train['target']==0].Char_length,kde=True,bins=20,hist=True,ax=axes[0],label="Histogram of 20 bins of label 0",

            kde_kws={"color": "r", "lw": 2, "label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})

axes[0].legend(loc="best")

sns.distplot(df_train[df_train['target']==1].Char_length,kde=True,bins=20,hist=True,ax=axes[1],label="Histogram of 20 bins of label 1",

            kde_kws={"color": "g", "lw": 2, "label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

axes[1].legend(loc="best")



plt.figure(figsize=(14,4))

sns.distplot(df_train[df_train['target']==0].Char_length,kde=True,bins=20,hist=True,label="Histogram of 20 bins of label 0",

            kde_kws={"color": "r", "lw": 2,"label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})



sns.distplot(df_train[df_train['target']==1].Char_length,kde=True,bins=20,hist=True,label="Histogram of 20 bins of label 1",

            kde_kws={"color": "g", "lw": 2,"label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

plt.legend(loc="best")
def word_count(sent):

    return len(sent.split())

df_train['word_count'] = df_train.text.apply(word_count)
df_train.head()
f, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

f.suptitle("Histogram of char length of text",fontsize=20)

sns.distplot(df_train[df_train['target']==0].word_count,kde=True,bins=20,hist=True,ax=axes[0],label="Histogram of label 0",

            kde_kws={"color": "r", "lw": 2, "label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})

axes[0].legend(loc="best")

sns.distplot(df_train[df_train['target']==1].word_count,kde=True,bins=20,hist=True,ax=axes[1],label="Histogram of label 1",

            kde_kws={"color": "g", "lw": 2, "label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

axes[1].legend(loc="best")



plt.figure(figsize=(14,4))

sns.distplot(df_train[df_train['target']==0].word_count,kde=True,bins=20,hist=True,label="Histogram of 20 bins of label 0",

            kde_kws={"color": "r", "lw": 2,"label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})



sns.distplot(df_train[df_train['target']==1].word_count,kde=True,bins=20,hist=True,label="Histogram of 20 bins of label 1",

            kde_kws={"color": "g", "lw": 2,"label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

plt.legend(loc="best")
def urls(sent):

    return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',sent)

def url_counts(sent):

    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',sent))

def remove_urls(sent):

    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',sent)

    
s ='hello this is the work - https://www.helloworld.com, https://www.worldhello.com'

print(urls(s))

print(url_counts(s))

print(remove_urls(remove_urls(s)))
%%time



df_train['url_count'] = df_train.text.apply(url_counts)

df_train['urls'] = df_train.text.apply(urls)
# An overview of dataframe after above transformations

df_train.head()
sum(df_train.url_count)
f, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

f.suptitle("Histogram of char length of text",fontsize=20)

sns.distplot(df_train[df_train['target']==0].url_count,kde=False,bins=10,hist=True,ax=axes[0],label="Histogram of label 0",

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})

axes[0].legend(loc="best")

sns.distplot(df_train[df_train['target']==1].url_count,kde=False,bins=10,hist=True,ax=axes[1],label="Histogram of label 1",

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

axes[1].legend(loc="best")



plt.figure(figsize=(14,4))

sns.distplot(df_train[df_train['target']==0].url_count,kde=False,bins=10,hist=True,label="Histogram of 10 bins of label 0",

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "y"})



sns.distplot(df_train[df_train['target']==1].url_count,kde=False,bins=10,hist=True,label="Histogram of 10 bins of label 1",

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "pink"})

plt.legend(loc="best")
df_train['text'] = df_train.text.apply(remove_urls)