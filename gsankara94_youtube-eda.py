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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
file = '../input/USvideos.csv'
df = pd.read_csv(file)
df.tail(n=10)
df.tail(n=2)
df.shape
# Let's study column datatypes
df.info()
# any NaN values?
df.isnull().sum()
# Let's study a summary of the dataset 
df.describe()
from scipy.stats import norm
views_log = np.log(df.views)
f,(ax1,ax2) = plt.subplots(1,2,figsize=(15,9))
sns.distplot(df['views'],hist=False,ax=ax1)
sns.distplot(views_log,hist=False,fit=norm,ax=ax2)
plt.show()
# Let's first study the number of times a channel has reached viral status 
most_popular = df['channel_title'].value_counts().sort_values(ascending=False) 
most_popular[:10]
from nltk.tokenize import word_tokenize
ESPN = df[df.channel_title == 'ESPN']
ESPN_words = [word_tokenize(sent) for sent in ESPN['title']] # We tokenize the sentneces into words first 
ESPN_words = [word for sublist in ESPN_words for word in sublist]  # We make the list flat out of list of sublists 
# Calculate frequency of words here
pd.Series(ESPN_words).value_counts().sort_values(ascending=False)[:10]
# First we can tokenize the words in each row. 
from nltk.tokenize import word_tokenize
words = [word_tokenize(sent) for sent in df['title']]
# Youtube Categories
import json
with open('../input/US_category_id.json') as json_data:
    d = json.load(json_data)
items = d.get('items')
ID = [item['id'] for item in items]
snippet = [item['snippet'] for item in items]
title =  [item['title'] for item in snippet]
Categories = dict(list(zip(ID,title)))
Categories = pd.Series(Categories, name='Categories').reset_index()
df['category_id'] = df['category_id'].astype('int64')
Categories['index'] = Categories['index'].astype('int64')
df_categories = df.merge(Categories, left_on='category_id', right_on='index',how='left')
plt.figure(figsize=(12,10))
df_categories['Categories'].value_counts().plot(kind='bar')
plt.show()
# Let's study the views by category (box-plot)
df_categories['log_likes'] = np.log(df_categories['likes'])
plt.figure(figsize=(16,8))
g=sns.boxplot(x='Categories',y='log_likes',data=df_categories)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
plt.show()
df_categories['log_dislikes'] = np.log(df_categories['dislikes'])
plt.figure(figsize=(16,8))
g=sns.boxplot(x='Categories',y='log_dislikes',data=df_categories)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
plt.show()
# Number of comments 
df_categories['log_comments'] = np.log(df_categories['comment_count'])
plt.figure(figsize=(16,8))
g=sns.boxplot(x='Categories',y='log_comments',data=df_categories)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
plt.show()
# Let's extract number of words, number of punctuations, number of capital words for each viral video title. 
import re
df_categories['word_count'] = df_categories['title'].apply(lambda x:len(x.split()))
pat = '[.!?:]+'
df_categories['punct_count'] = df_categories['title'].apply(lambda x: len(re.findall(pat, x)))
#All Cap Words
# Get another pattern to extract all capital words and count them
df_categories['capital_words'] = df_categories['title'].apply(lambda x: len(re.findall('[A-Z]+', x)))
# boxplot of word count for every category
plt.figure(figsize=(16,8))
h = sns.boxplot(x='Categories',y='word_count',data=df_categories)
h.set_xticklabels(h.get_xticklabels(),rotation=45)
plt.show()
plt.figure(figsize=(16,8))
h = sns.boxplot(x='Categories',y='capital_words',data=df_categories)
h.set_xticklabels(h.get_xticklabels(),rotation=45)
plt.show()
plt.figure(figsize=(16,8))
h = sns.boxplot(x='Categories',y='punct_count',data=df_categories)
h.set_xticklabels(h.get_xticklabels(),rotation=45)
plt.show()
# Now we should do a correlation matrix with the dataFrame

corr = df_categories[['log_likes','log_dislikes','views','word_count','punct_count','capital_words']].corr()
f,ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr,annot=True,ax=ax)
plt.show()
# convert trend time from object to datetime format 
import datetime
df_categories['trending_date'] = df_categories['trending_date'].apply(lambda x:datetime.datetime.strptime(str(x), '%y.%d.%m'))
#Convert publish_time into datetime format too 
df_categories['publish_time'] = df_categories['publish_time'].apply(lambda x:datetime.datetime.strptime(str(x),'%Y-%m-%dT%H:%M:%S.000Z' ))
# Now I want to find the number of days its been since the publish date and trend date
df_categories['days'] = abs(df_categories['publish_time']-df_categories['trending_date']).apply(lambda x:x.days)
# Now study the the days data 
df_categories.days.describe()
df_categories['tag_count'] = df_categories['tags'].apply(lambda x:len(re.findall('[|]',str(x))))
df_categories.tag_count = df_categories.tag_count.apply(lambda x:x+1)

# Tag count analysis by category
plt.figure(figsize=(16,8))
h = sns.boxplot(x='Categories',y='tag_count',data=df_categories)
h.set_xticklabels(h.get_xticklabels(),rotation=45)
plt.show()
corr = df_categories[['views','tag_count','likes','dislikes']].corr()
f,ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr,annot=True,ax=ax)
plt.show()