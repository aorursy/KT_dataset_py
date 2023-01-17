import numpy as np
import pandas as pd
import wordcloud
import functools
import nltk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plotly import express as px
%matplotlib inline
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
airbnb=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb.head(10)
# Find number of Rows
len(airbnb)
# Check the Data Types
airbnb.dtypes
# Check for Null Values
airbnb.isnull().sum()
# last review cl=olum may not prove to be useful for analysis, although reviews per month will, so replace NaN values in 'reviews_per_month'
airbnb.fillna({'reviews_per_month':0}, inplace=True)

# Also check the unique neighbourhoods
airbnb.neighbourhood_group.unique()
# What about unique Neighbourhoods?
airbnb.neighbourhood.unique()
# Check different Room Types
airbnb.room_type.unique()
fig, ax= plt.subplots(2,1, figsize=(20,10))

ax[0].set_title("Histogram of prices (total)")
sns.distplot(airbnb['price'], kde=False, ax=ax[0])
ax[1].set_title("Histogram of prices (< $500)")
sns.distplot(airbnb[airbnb['price'] < 500]['price'], kde=False, ax=ax[1])



fig.suptitle("Distribution of prices")
fig.show()
# Price Distribution across Neighbourhood Groups
fig, ax= plt.subplots(2,1, figsize=(20,20))

temp = airbnb.groupby(['neighbourhood_group'])['price'].mean().reset_index().sort_values('price',ascending=False)
ax[0].set_title("Prices Across Neighbourhood Groups")
sns.barplot(temp['neighbourhood_group'], temp['price'], ax=ax[0], palette=("Blues_d"))


temp = airbnb.groupby(['neighbourhood'])['price'].mean().reset_index().sort_values('price',ascending=False)
ax[1].set_title("Prices Across Neighbourhoods")
sns.barplot(temp['neighbourhood'][:20], temp['price'][:20], ax=ax[1], palette=("Blues_d"))
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=50, ha="right")

fig.suptitle("Distribution of prices")
fig.show()
# Using violinplot to showcase density and distribtuion of prices 
plt.figure(figsize=(15,15))
sns.violinplot(data=airbnb[airbnb['price'] < 500], x='neighbourhood_group', y='price')
plt.title('Density and distribution of prices for each neighberhood_group')
# Price Distribution across Room type
fig, ax= plt.subplots(1,2, figsize=(20,10))

temp = airbnb.groupby(['room_type'])['price'].mean().reset_index().sort_values('price',ascending=False)
sns.barplot(temp['room_type'], temp['price'], ax=ax[0], palette=("Blues_d"))
ax[0].set_title("Prices Across Room types")

sns.violinplot(data=airbnb[airbnb['price'] < 500], x='room_type', y='price', ax=ax[1], palette=("Blues_d"))
ax[1].set_title("Prices distibution Across Room types")

fig.show()
# Prices in Neighburhood Groups across Room Types 
f = sns.catplot(x="price", y="neighbourhood_group", hue="room_type", data=airbnb[airbnb['price'] <= 400], kind='violin', height=10, aspect=20/15)
f.axes[0][0].set_xlim(0,)
plt.title('Price distribution of each room type in each NYC Neighbourhood Groups')
airbnb['minimum_nights'].describe()
plt.figure(figsize=(20,10))
sns.distplot(airbnb[(airbnb['minimum_nights'] <= 30) & (airbnb['minimum_nights'] > 0)]['minimum_nights'], bins=31, kde=False)
temp = airbnb[airbnb['minimum_nights']<30].groupby(['minimum_nights'])['price'].median().reset_index()

plt.figure(figsize=(15,8))
sns.barplot(temp['minimum_nights'], temp['price'], palette=("Blues_d"))
plt.title('Mean price / minimum nights')
plt.xlabel('Minimum nights')
plt.ylabel('Price ($)')
top_host=airbnb.host_id.value_counts().head(10)
top_host
airbnb[airbnb['host_id'] == 219517861].head(1)
airbnb[airbnb['host_id'] == 107434423].head(1)
airbnb[airbnb['host_id'] == 30283594].head(1)
# Words that dont add value considering that data is from NYC
UNWANTED_WORDS = set(['manhattan', 'queen', 'brooklyn', 'nyc'])

fig, ax = plt.subplots(figsize=(12,8))
text = functools.reduce(lambda a,b: a + " " + str(b), airbnb.sample(frac=0.3)['name'])
text = ' '.join([w for w in nltk.word_tokenize(text) if w.lower() not in UNWANTED_WORDS])

wc = wordcloud.WordCloud(max_font_size=40).generate(text)
ax.imshow(wc, interpolation='bilinear')
ax.set_title("Most used words the dataset")
plt.axis("off")
fig.show()
top_reviewed_listings=airbnb.nlargest(10,'number_of_reviews')
top_reviewed_listings
# Average Price per night in top reviewed listings
price_avrg=top_reviewed_listings.price.mean()
print('Average price per night in top reviewed listings: $ {}'.format(price_avrg))
def get_top_terms(documents, ngram_range=(1,1), unwanted_words=set(), min_occurrences=1):
    """ Get a list of the most common n-grams (sorted)
    Params
    ----------
        documents: List of documents to analyze
        ngram_range (tuple): Whether extracting up to n-grams
        unwanted_words (set): Set of custom blacklist of words
        min_occurrences (int): return only words with occurrences >= min_occurrences
    Returns
    ----------
        List of tuples with (word, n_times).
    """
    blacklist = set(stopwords.words('english')).union(unwanted_words)
    vec = CountVectorizer(stop_words = blacklist, ngram_range=ngram_range)
    sum_words = vec.fit_transform(documents).sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    if min_occurrences > 1:
        words_freq = list(filter(lambda x: x[1] >= min_occurrences, words_freq))
    return words_freq
fig, ax= plt.subplots(5,1,figsize=(25,38))

top_ngrams = get_top_terms(airbnb.sort_values('reviews_per_month', ascending=False)['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)
sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[0],  palette=("Blues_d"))
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=40, ha="right")
ax[0].set_title('Most common n-grams of the top 200 popular rooms')


top_ngrams = get_top_terms(airbnb.sort_values('price', ascending=False)['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)
sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[1],  palette=("Blues_d"))
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=40, ha="right")
ax[1].set_title('Most common n-grams of the top 200 expensive rooms')


top_ngrams = get_top_terms(airbnb.sort_values('price', ascending=True)['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)
sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[2],  palette=("Blues_d"))
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=40, ha="right")
ax[2].set_title('Most common n-grams of the 200 cheapest rooms')


top_ngrams = get_top_terms(airbnb[airbnb['calculated_host_listings_count']>50]['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)
sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[3],  palette=("Blues_d"))
ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=40, ha="right")
ax[3].set_title('Most common n-grams of the top 200 professional rooms')


top_ngrams = get_top_terms(airbnb[airbnb['calculated_host_listings_count']<4]['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)
sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[4],  palette=("Blues_d"))
ax[4].set_xticklabels(ax[4].get_xticklabels(), rotation=40, ha="right")
ax[4].set_title('Most common n-grams of the top 200 amateur rooms')


fig.show()
viz_1=airbnb[airbnb['price']<600].plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
viz_1.legend()

import urllib
#initializing the figure size
plt.figure(figsize=(10,8))
#loading the png NYC image found on Google and saving to my local folder along with the project
i=urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')
nyc_img=plt.imread(i)
# scaling the image based on the latitude and longitude max and mins for proper output
# airbnb[airbnb['price']<600]['latitude'].max()
plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
airbnb[airbnb['price']<600].plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price', ax=ax,
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, zorder=5)
plt.legend()
plt.show()