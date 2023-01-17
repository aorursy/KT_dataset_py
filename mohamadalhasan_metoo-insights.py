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
#nltk.download('stopwords') #remove Comment tag and install this package

#nltk.download('wordnet') #remove Comment tag and install this package

import pandas

import re

import nltk



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sb

from os import path

from PIL import Image



import matplotlib.pyplot as plt

%matplotlib inline

#Load tweets to dataframe

df = pandas.read_csv('/kaggle/input/MeToo_tweets.csv')



# some exploration to our tweets

#Fetch how many word in each Tweet

df['word_count'] = df['Text'].apply(lambda x: len(str(x).split(" ")))

df[['Text','word_count']].head(2)
#Preliminary exploratiog text 

#Fetch wordcount for each Tweet

df['word_count'] = df['Text'].apply(lambda x: len(str(x).split(" ")))

df[['Text','word_count']].head(3)
df.word_count.describe()
#How many words in the dataset

df['word_count'].sum()
#get common words

#Series.value_counts() return a Series containing counts of unique values in descending order, Excludes NA values by default.

mfreq = pandas.Series(' '.join(df['Text']).split()).value_counts()[:25]

mfreq
#get uncommon words

unfreq =  pandas.Series(' '.join(df 

         ['Text']).split()).value_counts()[-25:]

unfreq


#import stop words from package and adding custom stopwords

stop_words = set(stopwords.words("english"))

#Creating a list of custom stopwords

new_words = ["http","using", "show", "result", "large", "also", "iv", "one", "two", "nshe","new", "previously", "shown","xa","xe","rt","oct","th","co","metoo","amp","ever"]

#Noise removal, Normalization(just removes the last few characters)

#Lemmatisation(is the process of converting a word to its base form)



#put each Tweet in  an array[i] in order to clean it(Noise Removal), 

corp = []

for i in range(0, df.shape[0]):#df.shape[0] get how many rows in dataset

    # 1Remove punctuations

  

    txt=re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",df['Text'][i])

    p=re.compile(r'\<http.+?\>', re.DOTALL)

    txt = re.sub(p, '', txt)

    

    # 2 -convert to lowercase

    txt = txt.lower()

    

    # 3 -remove tags

    txt=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",txt)

    

    # 4 -remove digits and

    txt=re.sub("(\\d|\\W)+"," ",txt)

    

    # 5 -Create a list from string

    txt = txt.split()

    

    # 6 -Doing Lemmatisation

    #Lemmatisation it's better than Stemming

    #Caring’ -> Lemmatization -> ‘Care’

    #‘Caring’ -> Stemming -> ‘Car’

    lemm = WordNetLemmatizer()

    txt = [lemm.lemmatize(word) for word in txt if not word in  

            stop_words] 

    txt = " ".join(txt)

    corp.append(txt)
#  plot word cloud image.

wordcloud = WordCloud(width=1600, height=800).generate(str(corp))

plt.figure( figsize=(20,10), facecolor='k')

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
#Creating a vector of word counts

#ignore terms that appear in more than 70% of the tweets

cvec=CountVectorizer(max_df=0.7,stop_words=stop_words, max_features=10000, ngram_range=(1,3))

Fit=cvec.fit_transform(corp)
#Most frequent single words after deleting stop words

def fetch_top_nwords(corp, n=None):

    vec = CountVectorizer().fit(corp)

    words_bag = vec.transform(corp)

    sum_words = words_bag.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in      

                   vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                       reverse=True)

    return words_freq[:n]

#Convert  freq words to dataset to plot bar plot

top_words = fetch_top_nwords(corp, n=25)

top_df = pandas.DataFrame(top_words)

top_df.columns=["Word", "Frequency"]

#plot  most freq words

sb.set(rc={'figure.figsize':(13,8)})

k = sb.barplot(x="Word", y="Frequency", data=top_df)

k.set_xticklabels(k.get_xticklabels(), rotation=30)
#Most frequent  Bi-grams

def fetch_top_n2words(corp, n=None):

    vecn2 = CountVectorizer(ngram_range=(2,2),  

            max_features=2000).fit(corp)

    words_bag = vecn2.transform(corp)

    sum_words = words_bag.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vecn2.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top_2words = fetch_top_n2words(corp, n=20)

top2_df = pandas.DataFrame(top_2words)

top2_df.columns=["Bi-gram", "Frequency"]

print(top2_df)

#Plot most freq Bi-grams



sb.set(rc={'figure.figsize':(13,8)})

b=sb.barplot(x="Bi-gram", y="Frequency", data=top2_df)

b.set_xticklabels(b.get_xticklabels(), rotation=45)
# Get most frequent Tri-grams

def fetch_top_n3words(corp, n=None):

    vecn3 = CountVectorizer(ngram_range=(3,3), 

           max_features=2000).fit(corp)

    words_bag = vecn3.transform(corp)

    sum_words = words_bag.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vecn3.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top_3words = fetch_top_n3words(corp, n=20)

top3_df = pandas.DataFrame(top_3words)

top3_df.columns=["Tri-gram", "Frequency"]

print(top3_df)

#Plot  most freq Tri-grams



sb.set(rc={'figure.figsize':(13,8)})

p=sb.barplot(x="Tri-gram", y="Frequency", data=top3_df)

p.set_xticklabels(p.get_xticklabels(), rotation=45)
#Get the most popular tweet in the dataset

df['Total']=df['Favorite_count']+df['Retweet_count']

Mot_Popular=df[df['Total']>=df['Total'].max()]

Mot_Popular[['Text','Total']].head(1)
#Show number of tweets by device/mobile or platform comming from

Source_Tweet = df.groupby("Source")

#Source_Tweet.count().sort_values(by="Source",ascending=False)

plt.figure(figsize=(15,10))

Source_Tweet.size().sort_values(ascending=False).head(10).plot.bar()

plt.xticks(rotation=50)

plt.xlabel("Number of Tweets by its Source")#.encode('utf-8')

plt.ylabel("")

plt.show()