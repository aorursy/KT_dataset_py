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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud,STOPWORDS

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize,sent_tokenize

from bs4 import BeautifulSoup

import re,string,unicodedata

from nltk.tokenize.toktok import ToktokTokenizer

from nltk.stem import PorterStemmer,WordNetLemmatizer

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

from string import punctuation

from nltk import pos_tag

from nltk.corpus import wordnet

import keras

from keras.models import Sequential
news = pd.read_csv("../input/cbc-news-coronavirus-articles-march-26/news.csv")
news.head()
news.authors.unique()
news.authors.replace("[]" , "Unknown"  ,inplace = True)
authors = news.authors.value_counts().index.values[:5]

freq = news.authors.value_counts().values[:5]

authors_pd = pd.DataFrame(columns = ['authors' , 'freq'])

authors_pd['authors'] = authors

authors_pd['freq'] = freq
f, ax = plt.subplots(figsize=(10, 10))

sns.barplot(x = "freq" , y = "authors" , data = authors_pd , label = 'Total')

sns.set_color_codes("muted")

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(ylabel="",xlabel="Maximum Number of Articles Written by Top 5 Authors")
news['Unnamed: 0'].count()
def sort_text(texts):

    des_text = news[news.text == texts].description.values[0]

    if(des_text.split()[:5] == texts.split()[:5]):

        return texts

    else:

        return texts +  ' ' + des_text

news.text = news.text.apply(sort_text)    
stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
def get_simple_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            pos = pos_tag([i.strip()])

            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))

            if(word.lower().isalpha()):

                final_text.append(word.lower())

    return " ".join(final_text)
news.text = news.text.apply(lemmatize_words)
plt.figure(figsize = (20,20))

wc = WordCloud(max_words = 3000 , width = 1600 , height = 800).generate(" ".join(news.text))

plt.imshow(wc , interpolation = 'bilinear')
cv=CountVectorizer(min_df=0,max_df=1,ngram_range=(1,3))

cv_reviews=cv.fit_transform(news.text)
cv_reviews
cv.vocabulary_
len(cv.vocabulary_)
d = cv.vocabulary_

gram = {}

bigram = {}

trigram = {}

for i in d.keys():

    x = i.split()

    if(len(x) == 1):

        gram[i] = gram.get(i,0) + d[i]

    elif(len(x) == 2):

        bigram[i] = bigram.get(i,0) + d[i]

    elif(len(x) == 3):

        trigram[i] = trigram.get(i,0) + d[i]     
# Sorting dictionaries based on values

gram = sorted(gram.items(),key = lambda kv:(kv[1], kv[0]))

bigram = sorted(bigram.items(),key = lambda kv:(kv[1], kv[0]))

trigram = sorted(trigram.items(),key = lambda kv:(kv[1], kv[0]))
gram_words = list(dict(gram).keys())[-20:]

gram_freq =  list(dict(gram).values())[-20:]



bigram_words = list(dict(bigram).keys())[-20:]

bigram_freq =  list(dict(bigram).values())[-20:]



trigram_words = list(dict(trigram).keys())[-20:]

trigram_freq =  list(dict(trigram).values())[-20:]
gram_freq_new = [i//100000 for i in gram_freq]

bigram_freq_new = [i//100000 for i in bigram_freq]

trigram_freq_new = [i//100000 for i in trigram_freq]
gram_df = pd.DataFrame(columns = ['words' , 'freq'])

gram_df['words'] = gram_words

gram_df['freq'] = gram_freq_new

f, ax = plt.subplots(figsize=(20, 20))

sns.barplot(x = "freq" , y = "words" , data = gram_df , label = 'Total')

sns.set_color_codes("muted")

ax.legend(ncol=1, loc="upper right", frameon=True)

ax.set(ylabel="",xlabel="Top 20 Grams used in Text in Lakhs")
bigram_df = pd.DataFrame(columns = ['words' , 'freq'])

bigram_df['words'] = bigram_words

bigram_df['freq'] = bigram_freq_new

f, ax = plt.subplots(figsize=(20, 20))

sns.barplot(x = "freq" , y = "words" , data = bigram_df , label = 'Total')

sns.set_color_codes("muted")

ax.legend(ncol=1, loc="upper right", frameon=True)

ax.set(ylabel="",xlabel="Top 20 BiGrams used in Text in Millions")
trigram_df = pd.DataFrame(columns = ['words' , 'freq'])

trigram_df['words'] = trigram_words

trigram_df['freq'] = trigram_freq

f, ax = plt.subplots(figsize=(20, 20))

sns.barplot(x = "freq" , y = "words" , data = trigram_df , label = 'Total')

sns.set_color_codes("muted")

ax.legend(ncol=1, loc="upper right", frameon=True)

ax.set(ylabel="",xlabel="Top 20 TriGrams used in Text in Millions")