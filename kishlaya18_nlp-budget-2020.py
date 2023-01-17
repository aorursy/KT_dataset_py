# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing Some Packages

import string

from string import punctuation

import nltk

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import preprocessing

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

from nltk.tokenize import wordpunct_tokenize

from nltk.stem import PorterStemmer , SnowballStemmer

stemmer_s = SnowballStemmer('english')

from string import punctuation

stop_nltk=stopwords.words("english")

from nltk.tokenize import TweetTokenizer

tw=TweetTokenizer()

from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
data=pd.read_csv('/kaggle/input/budget-txt/budget_speech.txt',error_bad_lines=False, warn_bad_lines=False)
txt = " ".join(data['Budget 2020-2021'].values)
txt[:40]
wordcloud = WordCloud().generate(txt)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud)

plt.show()
# A little Beautification.

word_cloud = WordCloud(width=800,height=800,background_color='white',

                      max_words=150).generate(txt)

plt.figure(figsize=(15,10))

plt.imshow(word_cloud)

plt.show()
#Getting words and its frequency in dict-key-values.

y = {} 

for i in txt.split(' '): 

    y[i] = y.get(i,0)+1

#conversion:

freq = {'words':list(y.keys()) , 'freq':list(y.values())}



mydata=pd.DataFrame(freq)

mydata.sort_values(by='freq',ascending=False).head()
#Applying tokenizer

text=word_tokenize(txt.lower()) # Converting all words to lower case for Uniform Casing.

print(len(txt),len(text))
fdist = FreqDist(text)

fdist.plot(30,cumulative=False)

plt.show()
from string import punctuation

stop_nltk=stopwords.words("english")

upd_stop = stop_nltk + ['department','government','proposed','under','centeral','will','ministry','provide','rate'] 

txt_upd = [term for term in text if term not in upd_stop and \

               term not in list(punctuation) and len(term)>2]

len(set(txt_upd))
newtxt=[lemm.lemmatize(word) for word in txt_upd]
#New WordCloud

mytxt=" ".join(newtxt)

#Initiating WordCloud

word_cloud = WordCloud().generate(mytxt)

#Beautifying

word_cloud = WordCloud(width=800,height=800,background_color='white',

                      max_words=150).generate(mytxt)

plt.figure(figsize=(15,10))

plt.imshow(word_cloud)

plt.show()
#Barplot

fdist = FreqDist(newtxt)

freq = {'words':list(fdist.keys()) , 'freq':list(fdist.values())}

df=pd.DataFrame(freq)

dat=df.sort_values(by='freq',ascending=False).head(25)

plt.figure(figsize=(10,12))

sns.barplot(data=dat,x='freq',y='words')

plt.show()
count_vect = CountVectorizer()

X = count_vect.fit_transform(newtxt)

#Getting the BOW:-

count_vect.get_feature_names()

#Getting the DataFrame

DTM = pd.DataFrame(X.toarray(),columns=count_vect.get_feature_names()) #datatermmatrix

# TDM=term document matrix

TDM = DTM.T

TDM.head()
#Bigrams

count_vect_bg = CountVectorizer(ngram_range=(2,2),max_features=25)

X_bg = count_vect_bg.fit_transform(newtxt)

DTM_bg = pd.DataFrame(X_bg.toarray(),columns=count_vect_bg.get_feature_names())

DTM_bg.drop(columns=['01 04','04 2020','19 20','2019 20','2020 21'],inplace=True)#Dont need date columns so dropping them.
plt.figure(figsize=(10,7))

x=DTM_bg.sum().sort_values(ascending=False).head(25)

y=x.reset_index()

y.head(1)

plt.pie(y[0],autopct='%1.1f%%', startangle=90, pctdistance=0.85,shadow=True)

plt.legend(y['index'],loc='lower right')

plt.show()
analyser = SentimentIntensityAnalyzer()

def get_vader_sentiment(sent):

    return analyser.polarity_scores(sent)['compound']

print(get_vader_sentiment(mytxt))
tfidf_vect =TfidfVectorizer()

X=tfidf_vect.fit_transform(newtxt)

X.toarray()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4,random_state=0)

y_means = kmeans.fit_predict(X.toarray())
BOW=tfidf_vect.get_feature_names()

num_clusters = 4

arr=kmeans.cluster_centers_

ordered_clu = arr.argsort()[:,::-1]

for i in range(num_clusters):

    print('Topics :',i)

    for i in ordered_clu[i,:5]:

        print(BOW[i])