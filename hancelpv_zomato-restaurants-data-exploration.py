# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from wordcloud import WordCloud, STOPWORDS
# Any results you write to the current directory are saved as output.
country_code = pd.read_excel('../input/Country-Code.xlsx')
country_code.columns = map(str.lower, country_code.columns)
country_code.columns = country_code.columns.str.replace('\s+', '_')
country_code.info()
zomato = pd.read_csv('../input/zomato.csv', encoding="ISO-8859-1")
zomato.columns = map(str.lower, zomato.columns)
zomato.columns = zomato.columns.str.replace('\s+', '_')
zomato.info()
# Joining to get country names
full = zomato.merge(country_code, on='country_code')
full.info()
full.groupby('country').count().loc[:, ['restaurant_id']].sort_values('restaurant_id', ascending=False)
# Filtering out Data for Indian Restaurants
data = full[full.country == 'India']
data.isnull().values.any()
data.describe()
col_list = ['average_cost_for_two', 'price_range', 'aggregate_rating', 'votes']

fig, axes = plt.subplots(1, 2)

data.average_cost_for_two.plot.hist(ax=axes[0], figsize=(16,7), bins=20)
axes[0].set_title('average_cost_for_two')

data.price_range.plot.hist(ax=axes[1], figsize=(16,7), bins=20)
axes[1].set_title('price_range')

fig, axes = plt.subplots(1, 2)

data.aggregate_rating.plot.hist(ax=axes[0], figsize=(16,7), bins=20)
axes[0].set_title('aggregate_rating')

data.votes.plot.hist(ax=axes[1], figsize=(16,7), bins=20)
axes[1].set_title('votes')
data.head()
wordcloud = WordCloud(background_color='white',width=800, height=400).generate(' '.join(data['city']))
plt.figure(figsize=(20,8))
plt.imshow(wordcloud)
plt.title('Word Cloud of Cities')
data['city'].value_counts().head().plot.barh(figsize=(12,5))
plt.title("Cities with most number of restaurants - TOP 5")
wordcloud = WordCloud(background_color='white',width=800, height=400).generate(' '.join(data['cuisines']))
plt.figure(figsize=(20,8))
plt.imshow(wordcloud)
plt.title('Word Cloud of cuisines')
data.has_online_delivery.value_counts().plot.bar(figsize=(10,6))
plt.title('Restaurant Distribution - Online Delivery')
data.head()
data[data.aggregate_rating > 0].plot(x='average_cost_for_two', y='aggregate_rating', kind='scatter', figsize=(12,6))
plt.title('Average Cost Vs Aggregate Rating')