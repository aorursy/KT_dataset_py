# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd 

from wordcloud import WordCloud, STOPWORDS

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import *

stop = set(stopwords.words("english"))

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
train = pd.read_table('../input/train.tsv', engine='c')

test = pd.read_table('../input/test.tsv', engine='c')
train.shape
test.shape
features=test.columns
features
train.describe()
test.describe()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=1000,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(train['item_description']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Train Item Description Word Cloud")
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=1000,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(test['item_description']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Test Item Description Word Cloud")
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(train['name']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Train Name Description Word Cloud")
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(test['name']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Test Name  Word Cloud")
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(train['brand_name']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Train Brand Name  Word Cloud")
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(test['brand_name']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Test Brand Name  Word Cloud")
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(train['category_name']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Train Category Name  Word Cloud")
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(test['category_name']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Test Category Name  Word Cloud")
#price distribution accross item condition

import seaborn as sns

sns.set(style="ticks")

sns.boxplot(x="item_condition_id", y="price", data=train, palette="PRGn")

sns.despine(offset=0, trim=True)
#price distribution accross shipping condition

import seaborn as sns

sns.boxplot(x="shipping", y="price", data=train, palette="PRGn")

sns.despine(offset=10, trim=True)
train.item_condition_id.replace(1,0,inplace=True)

train.item_condition_id.replace(2,0,inplace=True)

train.item_condition_id.replace(3,0,inplace=True)

train.item_condition_id.replace(4,1,inplace=True)

train.item_condition_id.replace(5,1,inplace=True)
#price distribution accross item condition

import seaborn as sns

sns.set(style="ticks")

sns.boxplot(x="item_condition_id", y="price", data=train, palette="PRGn")

sns.despine(offset=0, trim=True)
train[train['shipping']==0]['price'].median()
train[train['shipping']==1].price.median()
train['price'].hist(bins=25)
train.boxplot(column='price',figsize=(5,5),grid=True)
train.boxplot(column='price', by = 'item_condition_id')
train.boxplot(column='price', by = 'shipping')
train['brand_name'].mode()
test['brand_name'].mode()
train['category_name'].mode()
test['category_name'].mode()
train['item_description'].mode()
test['item_description'].mode()
train['name'].mode()
test['name'].mode()
train[train['name']=='Bundle'].count()
test[test['name']=='Bundle'].count()
train.name=train.name.str.lower()

train[train['name']=='bundle'].groupby(['category_name']).mean()
train[train['name']=='reserved'].groupby(['category_name']).mean()
train.price.plot(kind = 'hist',bins = 750,figsize = (15,15))
train.head()
train[(train.price > 2000)]
len(test[test.shipping==1])
len(test[test.shipping==0])