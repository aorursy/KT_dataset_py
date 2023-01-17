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
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.shape
test.shape
train.describe()
train.info()
train.head()
train['user_type'].unique()
len(train['user_id'].unique())
len(train['item_id'].unique())
len(train['region'].unique())
len(test['region'].unique())
len(train['city'].unique())
len(test['city'].unique())
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=400,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(train['description']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=400,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(test['description']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=400,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(train['title']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=400,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(test['title']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=400,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(train['category_name']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=400,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(test['category_name']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
