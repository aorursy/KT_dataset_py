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
from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#load dataset
df = pd.read_csv('../input/all.csv')
#first 10 rows of dataset
df.head(10)
#last 10 rows of dataset
df.tail(10)
#shape of dataset
df.shape
#info of dataset
df.info()
#description of dataset
df.describe()
#checking null values
df.isnull().sum()
#grouping dataset by type
df.groupby('type').count()
#grouping dataset by age
df.groupby('age').count()
#looking in content
df['content']
#Wordcloud for words in Mythology & Folklore type
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='lightsteelblue',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=15
                         ).generate(str(df[df['type']=='Mythology & Folklore']['content']))

fig = plt.figure(1,figsize=(14,20))
plt.imshow(wordcloud)
plt.title('Mythology & Folklore')
plt.axis('off')
plt.show()
#Wordcloud for words in Love type
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='bisque',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=15
                         ).generate(str(df[df['type']=='Love']['content']))

fig = plt.figure(1,figsize=(14,20))
plt.title('Love')
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#Wordcloud for words in Nature type
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='palegreen',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=15
                         ).generate(str(df[df['type']=='Nature']['content']))

fig = plt.figure(1,figsize=(14,20))
plt.title('Nature')
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#pie chart for types
labels = 'Love', 'Mythology & Folklore', 'Nature'
sizes = [326, 59, 188]


fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

plt.show()
#pie chart for ages
labels = 'Modern', 'Renaissance'
sizes = [258, 315]

fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

plt.show()

