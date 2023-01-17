# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk 

from nltk import word_tokenize

from nltk.corpus import stopwords

import re

import string

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/all.csv')
df.head()
def removePunctuation(x):

    x = x.lower()

    x = re.sub(r'[^\x00-\x7f]',r' ',x)

    x = x.replace('\r','')

    x = x.replace('\n','')

    x = x.replace('  ','')

    x = x.replace('\'','')

    return re.sub("["+string.punctuation+"]", " ", x)
stops = set(stopwords.words("english"))

def removeStopwords(x):

    filtered_words = [word for word in x.split() if word not in stops]

    return " ".join(filtered_words)
def processText(x):

    x= removePunctuation(x)

    x= removeStopwords(x)

    return x
df.groupby('type').count()
import numpy as np # linear algebra

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(df[df['type']=='Mythology & Folklore']['content']))



fig = plt.figure(1,figsize=(12,18))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(df[df['type']=='Love']['content']))



fig = plt.figure(1,figsize=(12,18))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(df[df['type']=='Nature']['content']))



fig = plt.figure(1,figsize=(12,18))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
labels = 'Love', 'Mythology & Folklore', 'Nature'

sizes = [326, 59, 188]

explode = (0, 0.2, 0)  # only "explode" the 2nd slice 



fig1, ax1 = plt.subplots(figsize=(10,10))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
len(df['author'].unique())
df['author'].value_counts().head(10)
df['age'].unique()
df['type'].unique()
from nltk.tokenize import sent_tokenize, word_tokenize

tin = pd.Series([word_tokenize(processText(x)) for x in df['content']])

tin.head(10)
from gensim.models import word2vec

num_features = 300    # Word vector dimensionality                      

min_word_count = 40   # Minimum word count                        

num_workers = 4       # Number of threads to run in parallel

context = 10          # Context window size                                                                                    

downsampling = 1e-3   # Downsample setting for frequent words

model = word2vec.Word2Vec(tin, workers=num_workers,size=num_features, 

                          min_count = min_word_count,

                          window = context, sample = downsampling)
model.most_similar('sun')