import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib import rcParams

import seaborn as sns

import nltk

from nltk.corpus import stopwords

stop = stopwords.words('english')

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from collections import Counter

import plotly.express as px

import plotly.figure_factory as ff

import re

import string



pd.options.display.max_columns = None



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = "/kaggle/input/tweet-sentiment-extraction"



train = pd.read_csv(os.path.join(path, "train.csv"))

test = pd.read_csv(os.path.join(path, "test.csv"))

sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))
train.head()
test.head()
sample_submission.head()
train.info()
train.describe()
test.info()
test.describe()
sns.set(style='darkgrid')



sns.countplot(data=train, x='sentiment', color="b")

plt.show()
sns.set(style='darkgrid')



sns.countplot(data=test, x='sentiment', color="b")

plt.show()
train['temp_list'] = train['selected_text'].apply(lambda x:str(x).split())

top = Counter([item for sublist in train['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
f, ax = plt.subplots(figsize=(6, 15))



sns.set_color_codes("pastel")

sns.barplot(x="count", y="Common_words", data=temp,

            label="Count", color="b")



ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, temp["count"].max()), ylabel="")

sns.despine(left=True, bottom=True)
def remove_stopword(text):

    return [w for w in text if not w in stop]



def clean_text(text):

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train['text'] = train['text'].apply(lambda x:clean_text(x))

train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))

train.drop(columns="temp_list", inplace=True)
train
train['temp_list'] = train['selected_text'].apply(lambda x:str(x).split())

train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))
train
top = Counter([item for sublist in train['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
f, ax = plt.subplots(figsize=(6, 15))



sns.set_color_codes("pastel")

sns.barplot(x="count", y="Common_words", data=temp,

            label="Count", color="b")



ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, temp["count"].max()), ylabel="")

sns.despine(left=True, bottom=True)
positive_train = train[train["sentiment"]=="positive"]

negative_train = train[train["sentiment"]=="negative"]

neutral_train = train[train["sentiment"]=="neutral"]
def tokenizeandstopwords(text):

    tokens = nltk.word_tokenize(text)

    token_words = [w for w in tokens if w.isalpha()]

    meaningful_words = [w for w in token_words if not w in stop]

    joined_words = ( " ".join(meaningful_words))

    return joined_words
positive_train["selected_text"] = positive_train["selected_text"].apply(clean_text)

negative_train["selected_text"] = negative_train["selected_text"].apply(clean_text)

neutral_train["selected_text"] = neutral_train["selected_text"].apply(clean_text)
positive_train["selected_text"] = positive_train["selected_text"].apply(tokenizeandstopwords)

negative_train["selected_text"] = negative_train["selected_text"].apply(tokenizeandstopwords)

neutral_train["selected_text"] = neutral_train["selected_text"].apply(tokenizeandstopwords)
positive_train['temp_list'] = positive_train['selected_text'].apply(lambda x:str(x).split())

positive_train['temp_list'] = positive_train['temp_list'].apply(lambda x:remove_stopword(x))

positive_top = Counter([item for sublist in positive_train['temp_list'] for item in sublist])

positive_temp = pd.DataFrame(positive_top.most_common(20))

positive_temp.columns = ['Common_words','count']

positive_temp.style.background_gradient(cmap='Blues')
negative_train['temp_list'] = negative_train['selected_text'].apply(lambda x:str(x).split())

negative_train['temp_list'] = negative_train['temp_list'].apply(lambda x:remove_stopword(x))

negative_top = Counter([item for sublist in negative_train['temp_list'] for item in sublist])

negative_temp = pd.DataFrame(negative_top.most_common(20))

negative_temp.columns = ['Common_words','count']

negative_temp.style.background_gradient(cmap='Blues')
neutral_train['temp_list'] = neutral_train['selected_text'].apply(lambda x:str(x).split())

neutral_train['temp_list'] = neutral_train['temp_list'].apply(lambda x:remove_stopword(x))

neutral_top = Counter([item for sublist in neutral_train['temp_list'] for item in sublist])

neutral_temp = pd.DataFrame(neutral_top.most_common(20))

neutral_temp.columns = ['Common_words','count']

neutral_temp.style.background_gradient(cmap='Blues')
positive_train['number of words'] = positive_train['text'].apply(lambda x : len(str(x).split()))

negative_train['number of words'] = negative_train['text'].apply(lambda x : len(str(x).split()))

neutral_train['number of words'] = neutral_train['text'].apply(lambda x : len(str(x).split()))



plt.figure(figsize=(15,8))

p1=sns.kdeplot(negative_train['number of words'], shade=True, color="r")

p1=sns.kdeplot(positive_train['number of words'], shade=True, color="b")

p1=sns.kdeplot(neutral_train['number of words'], shade=True, color="g")

p1.set_title('Distribution of Number Of words',fontsize=20)

plt.show()
from sklearn.feature_extraction.text import CountVectorizer





def WordRanking(corpus,n_gram,n=None):

   

    vec = CountVectorizer(ngram_range=n_gram,stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    

    return words_freq[:n]
positive_selected_text = train[train['sentiment']=='positive']['selected_text']

negative_selected_text = train[train['sentiment']=='negative']['selected_text']

neutral_selected_text = train[train['sentiment']=='neutral']['selected_text']



positive_bigrams = WordRanking(positive_selected_text,(2,2),20)

negative_bigrams = WordRanking(negative_selected_text,(2,2),20)

neutral_bigrams = WordRanking(neutral_selected_text,(2,2),20)



positive_bigrams = pd.DataFrame(positive_bigrams,columns=['word','counting'])

negative_bigrams = pd.DataFrame(negative_bigrams,columns=['word','counting'])

neutral_bigrams = pd.DataFrame(neutral_bigrams,columns=['word','counting'])
plt.figure(figsize=(19,10))

ax= sns.barplot(data=positive_bigrams,y='word',x='counting', color="b")

ax.set_title('Top 20 positive bigram words from selected text'.title(),fontsize=20)



ax.set_ylabel('Word counting',fontsize=15)

plt.show()
plt.figure(figsize=(19,10))

ax= sns.barplot(data=negative_bigrams,y='word',x='counting', color="b")

ax.set_title('Top 20 negative bigram words from selected text'.title(),fontsize=20)



ax.set_ylabel('Word counting',fontsize=15)

plt.show()
plt.figure(figsize=(19,10))

ax= sns.barplot(data=neutral_bigrams,y='word',x='counting', color="b")

ax.set_title('Top 20 neutral bigram words from selected text'.title(),fontsize=20)



ax.set_ylabel('Word counting',fontsize=15)

plt.show()