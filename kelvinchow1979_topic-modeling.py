# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#This notebook is not on predicting if a tweet is true. It is to practise on topic modeling on tweets. Thanks
#imports

import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns

import re

import string

from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import NMF

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
# read in the data set

data=pd.read_csv('../input/disaster-tweets/tweets.csv')
#preview the data set

data.head(5)
#A tweet example

data['text'][3]
# Remove, web link, numbers, punctuations, and to lower case

website= lambda x: re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", x)

alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)

punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())



data['reviews'] = data.text.map(website).map(alphanumeric).map(punc_lower)
#Taking only the True events

bag=data.loc[data['target']==1,'reviews']
#Checking the same tweet has been processed

bag[3]
# Using Count vectorizer with unigram

vectorizer1 = CountVectorizer(stop_words='english')

doc_word1 = vectorizer1.fit_transform(bag)

doc_word1.shape
#Check out the doc-word matrix

pd.DataFrame(doc_word1.toarray(), index=bag, columns=vectorizer1.get_feature_names()).head(2)
#Create a fucntion to display the topics and its word distributions

def display_topics(model, feature_names, no_top_words, topic_names=None):

    for ix, topic in enumerate(model.components_):

        if not topic_names or not topic_names[ix]:

            print("\nTopic ", ix)

        else:

            print("\nTopic: '",topic_names[ix],"'")

        print(", ".join([feature_names[i]

                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
#Latent Semantic Analysis with SVD

lsa1 = TruncatedSVD(10,random_state=123)

doc_topic1 = lsa1.fit_transform(doc_word1)

#lsa.explained_variance_ratio_,lsa.explained_variance_ratio_.sum()

display_topics(lsa1, vectorizer1.get_feature_names(), 10)
# Using Count vectorizer with bigram

vectorizer2 = CountVectorizer(ngram_range=(2,2), stop_words='english')

doc_word2 = vectorizer2.fit_transform(bag)
#Latent Semantic Analysis with SVD

lsa2 = TruncatedSVD(10,random_state=123)

doc_topics2 = lsa2.fit_transform(doc_word2)

#lsa.explained_variance_ratio_,lsa.explained_variance_ratio_.sum()

display_topics(lsa2, vectorizer2.get_feature_names(), 30)
#Explained variance ratio

lsa2.explained_variance_ratio_,lsa2.explained_variance_ratio_.sum()
#doc-term matrix

Vt2 = pd.DataFrame(doc_topics2.round(5),

             index = bag,

             columns = ["Thunderstorm","Tornado","Train Accident","Volcano Eruption","Sinkhole","Nuclear Meltdown","Hail Storm","Virus OutBreak","Terrorist Bombing","Electrocution" ])

Vt2
#Tag each tweet with a topic

Vt2['Cat']=Vt2.idxmax(axis=1, skipna=True)
#Make a count of tweets in each topic

count=pd.DataFrame(Vt2.Cat.value_counts().sort_values(ascending=False))

count.rename(columns={'Cat':'Tweets Count'},inplace=True)

count
#Plot it

sns.set_color_codes("pastel")

sns.barplot(x=count['Tweets Count'], y=count.index, data=count, orient='h', color="b",)

#plt.savefig('hbar.png')
# Using Count vectorizer with trigram

vectorizer3 = CountVectorizer(ngram_range=(3,3), stop_words='english')

doc_words3 = vectorizer3.fit_transform(bag)
#Latent Semantic Analysis with SVD

lsa3 = TruncatedSVD(8,random_state=123)

doc_topics3 = lsa3.fit_transform(doc_words3)

#lsa.explained_variance_ratio_,lsa.explained_variance_ratio_.sum()

display_topics(lsa3, vectorizer3.get_feature_names(), 10)
#Unigram and trigram do not give some distinction of topic, but bigram does, and with k = 10 because anything more are just repetive.