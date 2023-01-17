import numpy as np 

import pandas as pd

import seaborn as sns

import random

import json

import os

import wordcloud

import functools

import nltk

import requests



from matplotlib import pyplot as plt

from urllib.request import urlopen

from plotly import graph_objects as go

from plotly import express as px

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

from geopy.distance import geodesic #Distances
# A mapbox token is needed for some interactive map plots:

#          https://docs.mapbox.com/help/how-mapbox-works/access-tokens/

#mapbox_token = os.environ['MAPBOX_TOKEN'] #Set a public mapbox token as a string or environment variable.

mapbox_token = requests.get('https://pastebin.com/raw/GygwE5aD').text # Temporary token share (will be deleted)

px.set_mapbox_access_token(mapbox_token)
print("Data files:")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print("\t"+os.path.join(dirname, filename))



data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head(3)
print(f"Number of records: {data.count()[0]}")
#Fix NaN in reviews per month

data[data['reviews_per_month'].isna()] = 0



#Remove "empty records"

data = data[data['room_type'] != 0]
print(f"Number of records after cleaning: {data.count()[0]}")
fig, ax= plt.subplots(2,1, figsize=(20,10))



ax[0].set_title("Histogram of prices (total)")

sns.distplot(data['price'], kde=True, ax=ax[0])

ax[1].set_title("Histogram of prices (< $800)")

sns.distplot(data[data['price'] < 800]['price'], kde=True, ax=ax[1])







fig.suptitle("Distribution of prices")

fig.show()
d = data.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)



fig, ax= plt.subplots(2,1, figsize=(20,19))

sns.barplot(d.index.tolist()[:20], d.values[:20], ax=ax[0], palette=("Blues_d"))

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=40, ha="right")

ax[0].set_title('Top 20 most expensive neighbourhoods')



d = data.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)[:5]

sns.barplot(d.index.tolist(), d.values, ax=ax[1], palette=("Blues_d"))

#ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=40, ha="right")

ax[1].set_title('Mean price of each neighbourhood_group')



fig.show()

cscale = [

          [0.0, 'rgb(165,0,38)'], 

          [0.0005, 'rgb(215,48,39)'], 

          [0.007, 'rgb(250, 152, 122)'], 

          [0.08, 'rgb(208, 254, 144)'], 

          [0.1, 'rgb(0, 255, 179)'], 

          [0.3, 'rgb(171,217,233)'], 

          [0.7, 'rgb(116,173,209)'], 

          [0.9, 'rgb(69,117,180)'], 

          [1.0, 'rgb(49,54,149)']

         ]
fig = px.scatter_mapbox(data, lat="latitude", lon="longitude",  color="price", size="reviews_per_month",

                  color_continuous_scale=cscale, size_max=20, height=760, zoom=10, title='Scatter map of all Airbnb rents (price <= $700)', range_color=(0,700))

fig.show()
fig = px.scatter_mapbox(data[data['price'] > 700], lat="latitude", lon="longitude",  color="price", size="reviews_per_month",

                  color_continuous_scale=cscale, height=760,size_max=20, zoom=10, title='Scatter map of most expensive Airbnb rents (>$700)')

fig.show(renderer='kaggle')
d = data.groupby('room_type')['price'].mean().sort_values(ascending=False)

fig, ax= plt.subplots(figsize=(15,7))

sns.barplot(d.index.tolist(), d.values, ax=ax)

ax.set_title('Average price of room types')

fig.show()
fig, ax= plt.subplots(figsize=(20,8))

sns.violinplot(x="room_type", y="price", data=data[data['price'] <= 700], ax=ax)

ax.set_title('Price distribution of each room type')

fig.show()
f = sns.catplot(x="price", y="neighbourhood_group", hue="room_type", data=data[data['price'] <= 900], kind='violin', height=10)

f.axes[0][0].set_xlim(0,)

plt.title('Price distribution of each room type in each NYC zone')

plt.show()
fig = px.scatter_mapbox(data[data['price'] <= 700], lat="latitude", lon="longitude",  color="room_type", size="price",

                  color_continuous_scale=cscale, height=760,size_max=20, zoom=10, title='Type of room distribution on the map.')

fig.show(renderer='kaggle')
px.scatter(data[data['price']<700], x="reviews_per_month", y='price', color='room_type', title='Popularity/Price')

#sns.scatterplot(x="reviews_per_month", y='price', hue='room_type', data=data)
sns.pairplot(data, y_vars=['price'], x_vars=['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365'], height=15, hue='neighbourhood_group')
data['minimum_nights'].describe()
print(f'Percentile 95 of minimum_nights: {np.percentile(data["minimum_nights"], 95)}')

print(f'Mean of minimum_nights: {np.mean(data["minimum_nights"])}')

print(f"Mode of minimum_nights: {data[(data['minimum_nights'] <= 30) & (data['minimum_nights'] > 0)]['minimum_nights'].mode()[0]}")
sns.distplot(data[(data['minimum_nights'] <= 30) & (data['minimum_nights'] > 0)]['minimum_nights'], bins=31)
d = data[data['minimum_nights'] < 30].groupby('minimum_nights')['price'].median()
fig = plt.figure(figsize=(10,6))

plt.xticks(np.arange(min(d.keys().tolist()), max(d.keys().tolist())+1, 1.0))

plt.bar(d.keys().tolist(), d.tolist())

plt.title('Mean price / minimum nights')

plt.xlabel('Minimum nights')

plt.ylabel('Price ($)')

plt.grid(True)
host_listings_count = data.groupby('host_id')['calculated_host_listings_count'].mean()
summary = host_listings_count.describe()

summary
fig, ax = plt.subplots(1,2, figsize=(22,6))

sns.distplot(host_listings_count[host_listings_count < 10], kde=False, hist=True, ax=ax[0])

ax[0].set_title("Number of postings/user distribution (zoomed)")

sns.distplot(host_listings_count, kde=False, hist=True, ax=ax[1])

ax[1].set_title("Number of postings/user distribution (total)")

fig.show()
user_outliers = host_listings_count[host_listings_count >= (summary['mean'] + 2*summary['std'])]

user_outliers
summary
print("Users with most postings")

user_outliers.sort_values().tail(10)
data[data['host_id'] == 219517861].head(1)
data[data['host_id'] == 107434423].head(1)
data[data['host_id'] == 30283594].head(1)
posting_outliers = data[data['host_id'].isin(user_outliers.index.tolist())]

posting_normal = data[~data['host_id'].isin(user_outliers.index.tolist())]
palette ={"Entire home/apt":"C0","Private room":"C1","Shared room":"C2"}



fig, ax = plt.subplots(1,2, figsize=(26,10))



sns.violinplot(x="price", y="neighbourhood_group", hue="room_type", data=posting_outliers, ax=ax[0], 

               palette=palette, 

               order=['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx'], 

               hue_order=['Entire home/apt', 'Private room', 'Shared room'])

ax[0].set_title("Price distribution of each room type in each NYC zone (outliers/professionals)")

ax[0].set_xlim(0,250)



sns.violinplot(x="price", y="neighbourhood_group", hue="room_type", data=posting_normal[posting_normal['price'] <= 900], ax=ax[1], 

               palette=palette, 

               order=['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx'], 

               hue_order=['Entire home/apt', 'Private room', 'Shared room'])

ax[1].set_title("Price distribution of each room type in each NYC zone (amateurs)")

ax[1].set_xlim(0,250)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scattermapbox(

        lat=posting_outliers.latitude,

        lon=posting_outliers.longitude,

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=5,

            color='rgb(121,181,237)',

            opacity=0.7

        ),name='Professional'

    ))



fig.add_trace(go.Scattermapbox(

        lat=posting_normal.latitude,

        lon=posting_normal.longitude,

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=4,

            color='rgb(237,138,121)',

            opacity=0.2

        ), name='Amateur'

    ))



fig.update_layout(

    title='Professional / Amateur postings',

    autosize=True,

    hovermode='closest',

    showlegend=True,

    height=900,

    mapbox=go.layout.Mapbox(

        accesstoken=mapbox_token,

        bearing=0,

        center=go.layout.mapbox.Center(

            lat=40.74767,

            lon=-73.97500

        ),

        pitch=0,

        zoom=12,

        style='light'

    ),

)



fig.show(renderer='kaggle')
fig = px.scatter_mapbox(posting_outliers[posting_outliers['host_id'].isin(user_outliers.sort_values(ascending=False).head(20).index.tolist())], 

                        lat="latitude", lon="longitude",  color="host_name",

                  color_continuous_scale=cscale, height=760, zoom=12, title='Listings of the top 20 users', range_color=(0,700))

fig.show()
fig = px.scatter_mapbox(data[data['host_id'].isin(host_listings_count[host_listings_count == 2].head(20).index.tolist())], 

                        lat="latitude", lon="longitude",  color="host_name",

                  color_continuous_scale=cscale, height=760, zoom=12, title='20 users with 2 rooms posted', range_color=(0,700))

fig.show()
def get_nearest_neighbours(point, coords, treshold=100):

    """Given a list of coordinates, return the ones which are near to point.

    

    Args

    -------

    point (Tuple): with Lat/Long

    coords (List[Tuple]): latitudes/longitudes

    treshold: number of meters to consider a certain coordinate near to 'point'

    

    Return

    -----------

    nearest: List[int] indices of points near to point. 

    """

    nearest = []

    for idx, c in enumerate(coords):

        if geodesic(point, c).meters <= treshold:

            nearest.append(idx)

    return nearest



def calculate_clusters(coordinates, subsample=0.5):

    """ Given a list of coordinates (Lat/Long), return a estimation of the distribution of the number of clusters

    

    Args

    --------

    coordinates (NP Array of tuples): List of coordinates to analyze

    subsample (float): Percentage of number of points used for evaluation

    

    Return

    ------------

    ret: Tuple with mean and sd of the distribution

    """

    #coordinates = np.array([tuple(x) for x in posting_outliers[posting_outliers['host_name']==name][['latitude', 'longitude']].values])

    mask_test = np.random.choice(a=[False, True], size=coordinates.shape[0], p=[1-subsample, subsample])

    

    nb_neighs = []

    for idx, p in enumerate(coordinates):

        #p.delete(a, ind, axis=0)       

        nb_neighs.append(len(get_nearest_neighbours(p, np.delete(coordinates,idx,axis=0))))

    

    return np.mean(nb_neighs), np.std(nb_neighs)
#Testing the functions.

# Get all the rooms of a particular user (which is in the top room holders)

p = np.array([tuple(x) for x in posting_outliers[posting_outliers['host_name']=='John'][['latitude', 'longitude']].values])

#Get the mean number of points in the clusters and their standard dev

calculate_clusters(p)
print("Number of rooms per cluster of users with less than postings (mean): ")

test_list = host_listings_count[host_listings_count <= 5].head(100).index.tolist()

metrics = []

for user_id in test_list:

    p = np.array([tuple(x) for x in data[data['host_id']==user_id][['latitude', 'longitude']].values])

    metrics.append(calculate_clusters(p))

metrics = pd.DataFrame(metrics, columns=['mean', 'sd'])

metrics.mean()
print("Number of rooms per cluster of the top room holders: ")

test_list = user_outliers.sort_values(ascending=False).head(20).index.tolist()

metrics = []

for user_id in test_list:

    p = np.array([tuple(x) for x in data[data['host_id']==user_id][['latitude', 'longitude']].values])

    metrics.append(calculate_clusters(p))

metrics = pd.DataFrame(metrics, columns=['mean', 'sd'])

metrics.mean()
# Words that won't add anything apart from what we already know from the other data.

UNWANTED_WORDS = set(['manhattan', 'queen', 'brooklyn', 'nyc'])
fig, ax = plt.subplots(figsize=(12,8))



text = functools.reduce(lambda a,b: a + " " + str(b), data.sample(frac=0.3)['name'])

text = ' '.join([w for w in nltk.word_tokenize(text) if w.lower() not in UNWANTED_WORDS])



wc = wordcloud.WordCloud(max_font_size=40).generate(text)

ax.imshow(wc, interpolation='bilinear')

ax.set_title("Most used words the dataset")

plt.axis("off")

fig.show()
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
top_ngrams = get_top_terms(data.sort_values('reviews_per_month', ascending=False)['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)

fig, ax= plt.subplots(5,1,figsize=(25,38))

sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[0],  palette=("Blues_d"))

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=40, ha="right")

ax[0].set_title('Most common n-grams of the top 200 popular rooms')





top_ngrams = get_top_terms(data.sort_values('price', ascending=False)['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)

sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[1],  palette=("Blues_d"))

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=40, ha="right")

ax[1].set_title('Most common n-grams of the top 200 expensive rooms')





top_ngrams = get_top_terms(data.sort_values('price', ascending=True)['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)

sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[2],  palette=("Blues_d"))

ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=40, ha="right")

ax[2].set_title('Most common n-grams of the 200 cheapest rooms')





top_ngrams = get_top_terms(data[data['calculated_host_listings_count']>50]['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)

sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[3],  palette=("Blues_d"))

ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=40, ha="right")

ax[3].set_title('Most common n-grams of the top 200 professional rooms')





top_ngrams = get_top_terms(data[data['calculated_host_listings_count']<4]['name'].iloc[:200], ngram_range=(1,2), unwanted_words=UNWANTED_WORDS, min_occurrences=10)

sns.barplot([x[0] for x in top_ngrams], [x[1] for x in top_ngrams], ax=ax[4],  palette=("Blues_d"))

ax[4].set_xticklabels(ax[4].get_xticklabels(), rotation=40, ha="right")

ax[4].set_title('Most common n-grams of the top 200 amateur rooms')





fig.show()