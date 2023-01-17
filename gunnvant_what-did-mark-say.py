import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import nltk.corpus as corpus

import nltk
data=pd.read_csv("../input/mark.csv")
## Let's find out hou long were Mark's Responses?

def get_count(x):

    return len(nltk.word_tokenize(x))

data['len']=data['Text'].map(get_count)
data.head()
print("The total words spoken by Mark were {} words".format(data.query("Person=='ZUCKERBERG:'")['len'].sum()))

print("The average length of his response weas {} words".format(round(data.query("Person=='ZUCKERBERG:'")['len'].mean(),2)))

print("The maximum length of Mark's response was {} words".format(data.query("Person=='ZUCKERBERG:'")['len'].max()))

data.query("Person !='ZUCKERBERG:'").groupby("Person").sum().rename(columns={'len':'Total Words'}).sort_values("Total Words",ascending=False).head(20).plot(kind="barh",colormap="Set2",figsize=(8,8))

plt.title("Total Words Spoken",fontsize=30)

plt.ylabel("Senator",fontsize=25)

plt.yticks(fontsize=15)

plt.xlabel("Count",fontsize=15)
##  Most commonly used words by Mark

from sklearn.feature_extraction import text

def get_imp(bow,mf,ngram):

    tfidf=text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')

    matrix=tfidf.fit_transform(bow)

    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)
mark=data[data['Person']=="ZUCKERBERG:"]['Text'].tolist()

get_imp(mark,mf=5000,ngram=2).head(10)
get_imp(mark,mf=5000,ngram=3).head(10)
def get_imp_tf(bow,mf,ngram):

    tfidf=text.TfidfVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')

    matrix=tfidf.fit_transform(bow)

    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)
get_imp_tf(mark,mf=5000,ngram=2).head(10)
get_imp_tf(mark,mf=5000,ngram=3).head(10)