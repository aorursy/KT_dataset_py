# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_imdb = pd.read_csv("../input/imdb_labelled.txt", sep="\t", header=None)

df_amazon = pd.read_csv("../input/amazon_cells_labelled.txt", sep="\t", header=None)

df_yelp = pd.read_csv("../input/yelp_labelled.txt", sep="\t", header=None)
df_imdb.head(), df_amazon.head(), df_yelp.head()
df_corpus = pd.concat([df_amazon, df_imdb, df_yelp], axis=0)

df_amazon.shape, df_imdb.shape, df_yelp.shape, df_corpus.shape
from wordcloud import WordCloud

import matplotlib.pyplot as plt



def gen_wordcloud(data, max_words, stopwords):

    wordcloud = WordCloud(

        background_color = 'white',

        max_words = max_words,

        stopwords = stopwords,

        max_font_size = 40, 

        scale = 3,

        random_state = 1

    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))

    plt.axis('off')

    plt.imshow(wordcloud)

    plt.show()
stopwords = set(["I",

"a",

"about",

"an",

"are",

"as",

"at",

"be",

"by",

"com",

"for",

"from",

"is",

"it",

"of",

"on",

"or",

"that",

"this",

"was",

"what",

"when",

"will",

"with",

"how",

"in",

"the",

"to",

"the",

"www",

"where",

"who"])
from textblob import TextBlob

df_corpus['blobbed'] = df_corpus[0].apply(lambda x: TextBlob(x).words)
df_corpus.head()
df_corpus['cleaned'] = df_corpus['blobbed'].apply(lambda x: list(set(x.lower()) - stopwords))
df_corpus.head()
gen_wordcloud(df_corpus['cleaned'][df_corpus[1] == 0] , 200, stopwords)
negative_words =  ["n't", "not", "off", "but", "problem", "disappointed", "bad", "tied", "should", "no", "only", "waste", "con", "unless", "misleading", "suck", "wasted", "breakage", "little"]
gen_wordcloud(df_corpus['cleaned'][df_corpus[1] == 1] , 200, stopwords)
positive_words = ["excellent","pretty", "fine", "love", "great", "reasonable", "nice", "good", "impressed", "works", "must", "everything", "highly", "well", "value", "absolutely", "best", "ideal", "quality", "real", "helped", "happy", "delightful", 'pleasure', 'flavurful', 'wow']
def count_words(x, word_list):

    temp = []

    for word in word_list:

        if word in x:

            temp.append(word)

    return temp
df_corpus['pos_words'] = df_corpus['cleaned'].apply(lambda x: len(count_words(x, positive_words)))

df_corpus['neg_words'] = df_corpus['cleaned'].apply(lambda x: len(count_words(x, negative_words)))
df_corpus.head()
import math

df_corpus['polarity'] = df_corpus[0].apply(lambda x: TextBlob(x).sentiment.polarity)

df_corpus['polarity_norm'] = df_corpus[0].apply(lambda x: math.ceil(TextBlob(x).sentiment.polarity))

df_corpus.head()
df_corpus['polarity_norm'].value_counts()
df_corpus=df_corpus.replace({'polarity_norm': {-1: 0}}) 

df_corpus['polarity_norm'].value_counts()
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(df_corpus[1], df_corpus['polarity_norm']).ravel()

tp, tn, fp, fn 
#Recall and precision

recall = tp / (fn + tp)

precision = tp / (fp + tp)

recall, precision
def sentiment_process(pos, neg, tblob):

    if pos > neg:

        return 1

    elif pos < neg:

        return 0

    else:

        return tblob
df_corpus['sentiment_pred'] = df_corpus.apply(lambda x: sentiment_process(x['pos_words'], x['neg_words'], x['polarity_norm']) , axis=1)

df_corpus['sentiment_pred'].value_counts()
df_corpus.head(10)
tn, fp, fn, tp = confusion_matrix(df_corpus[1], df_corpus['sentiment_pred']).ravel()

tp, tn, fp, fn 
#Recall and precision

recall = tp / (fn + tp)

precision = tp / (fp + tp)

recall, precision