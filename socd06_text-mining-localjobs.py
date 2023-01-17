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
import pandas as pd

file_name = "/kaggle/input/ltj_halloween2019.csv"

df = pd.read_csv(file_name)
df.head()

#overview of data... needs edition
#editing dataframe

df=df.rename(columns={"Current Tech Opportunities in London, Ontario": "jobs"})

df=df.drop([0,1,2]) #removing first three data points as they are introducing noise

df=df["jobs"]

#restoring indexes since we deleted some

df=df.reset_index(drop=True)

df
#Separating words stuck to each other 

#run this twice or as neccesary to improve the data

for j in range(3):

    import re

    for i in range(len(df)):

        #identifying capital letters using regular expressions

        s = re.search(r'[a-z][a-z][A-Z]', df[i])

        #print(s)

        try:

            df[i] = df[i].replace(df[i][s.start()+2],("  "+df[i][s.start()+2]))                

        except:

            continue

df
# Libraries for text preprocessing

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

#nltk.download('wordnet') 

from nltk.stem.wordnet import WordNetLemmatizer
#Detected more noise in form of links in data

#df.replace(to_replace=r'http*')

string="Specialisthttps://www.linkedin.com/jobs/view/1555270712/"

s2 = re.search(r'http.*', string)

print(s2)

#print(s2.start())

#print(string[s2.start()::])

print(s2.group(0))

#removing any links

df=df.replace(to_replace=r'http.*',value=" ",regex=True)
#removing not alphanumeric characters

df=df.replace(to_replace=r'[^A-Za-z0-9]+',value=" ",regex=True)
#consolidating Sr as Senior and Jr as Junior

df=df.replace(to_replace=r'Sr',value="Senior",regex=True)

df=df.replace(to_replace=r'Jr',value="Junior",regex=True)
#Identify common words

freq_c = pd.Series(' '.join(df).split()).value_counts()[:50]

freq_c
#Identify uncommon words

freq_u =  pd.Series(' '.join(df).split()).value_counts()[-50:]

freq_u
##Creating a list of stop words

stop_words = set(stopwords.words("english"))

#could add 75K, 65K and 12 (as in 12 months) but will to keep it to see if they weigh on the data

#skipping lemmatisation
#Word cloud

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

%matplotlib inline



corpus1 = "".join(df[::])



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stop_words,

                          max_words=30,

                          max_font_size=50, 

                          random_state=42

                         ).generate(str(corpus1))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=900)
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))

X=cv.fit_transform(df)
#Most frequently occuring words

def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(df)

    bag_of_words = vec.transform(df)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in      

                   vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                       reverse=True)

    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot

top_words = get_top_n_words(df, n=20)

top_df = pd.DataFrame(top_words)

top_df.columns=["Word", "Freq"]

#Barplot of most freq words

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

g = sns.barplot(x="Word", y="Freq", data=top_df)

g.set_xticklabels(g.get_xticklabels(), rotation=30)
#Most frequently occuring Bi-grams

def get_top_n2_words(df, n=None):

    vec1 = CountVectorizer(ngram_range=(2,2),  

            max_features=2000).fit(df)

    bag_of_words = vec1.transform(df)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top2_words = get_top_n2_words(df, n=20)

top2_df = pd.DataFrame(top2_words)

top2_df.columns=["Bi-gram", "Freq"]

print(top2_df)

#Barplot of most freq Bi-grams

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)

h.set_xticklabels(h.get_xticklabels(), rotation=45)
#TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))

#tdm.trigram = TermDocumentMatrix(corpus.ng,

#control = list(tokenize = TrigramTokenizer))
#Most frequently occuring Tri-grams

def get_top_n3_words(df, n=None):

    vec1 = CountVectorizer(ngram_range=(3,3), 

           max_features=2000).fit(df)

    bag_of_words = vec1.transform(df)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top3_words = get_top_n3_words(df, n=20)

top3_df = pd.DataFrame(top3_words)

top3_df.columns=["Tri-gram", "Freq"]

print(top3_df)

print(type(top3_df))

#Barplot of most freq Tri-grams

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)

j.set_xticklabels(j.get_xticklabels(), rotation=45)
print(top3_df)

print(type(top3_df))
corpus2 = top2_df.set_index('Bi-gram')['Freq'].to_dict()

print(corpus2)

print(type(corpus2))
#Word cloud using the i-gram data

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stop_words,

                          max_words=10,

                          max_font_size=50, 

                          random_state=42

                         ).generate_from_frequencies(corpus2)

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word2.png", dpi=900)
corpus3 = top3_df.set_index('Tri-gram')['Freq'].to_dict()

print(corpus3)

print(type(corpus3))
#Word cloud using the tri-gram data

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stop_words,

                          max_words=10,

                          max_font_size=50, 

                          random_state=42

                         ).generate_from_frequencies(corpus3)

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word3.png", dpi=900)
#Inspired by https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34