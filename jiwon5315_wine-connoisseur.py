import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import sys

import os
df = pd.read_csv("../input/winemag-data-130k-v2.csv",sep = ",")

df.head(5)
columns = ['country', 'description', 'points', 'price', 'province', 'taster_name', 'title', 'variety', 'winery']

wine_reviews = df[columns].dropna(axis=0, how='any')

wine_reviews.info()
wine_reviews.head(5)
fig = plt.figure(figsize=(16,4))

sns.scatterplot(x='points', y='price', hue='price', data=wine_reviews)
wine_reviews.corr()
# Highest priced wine

wine_reviews.loc[wine_reviews['price']==wine_reviews.price.max()]
# Lowest priced wine

min_price = wine_reviews.loc[wine_reviews['price']==wine_reviews.price.min()]

print(min_price.mean())

min_price
print("Max Points:", wine_reviews.points.max())

print("There are", len(wine_reviews.loc[wine_reviews['points']==100]), "wines with the highest point of 100.")

print("Min Points:", wine_reviews.points.min())

print("There are", len(wine_reviews.loc[wine_reviews['points']==80]), "wines with the lowest point of 80.")
max = wine_reviews.loc[wine_reviews['points']==100]

min = wine_reviews.loc[wine_reviews['points']==80]



fig = plt.figure(figsize=(16,4))

fig.suptitle("Count of Price Ranges For Wines with Lowest Points (80)")

sns.countplot(min['price'], palette='spring')

fig2 = plt.figure(figsize=(16,4))

fig2.suptitle("Count of Price Ranges For Wines with Highest Points (100)")

sns.countplot(max['price'], palette='summer')
# Review sample

wine_reviews['description'].values[0]
# Lower-case wine reviews

wine_reviews['description'] = wine_reviews['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))

wine_reviews['description'].head(5)
# Remove punctuations from wine reviews

wine_reviews['description'] = wine_reviews['description'].str.replace('[^\w\s]','')

wine_reviews['description'].head()
# Remove stop words from wine reviews

from nltk.corpus import stopwords

stop = stopwords.words('english')

wine_reviews['description'] = wine_reviews['description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

wine_reviews['description'].head()
# Remove common words from wine reviews

freq = pd.Series(' '.join(wine_reviews['description']).split()).value_counts()[:10]

freq
# Remove common words 

freq = list(freq.index)

wine_reviews['description'] = wine_reviews['description'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

wine_reviews['description'].head()
# Spell-check wine reviews

from textblob import TextBlob

wine_reviews['description'][:5].apply(lambda x: str(TextBlob(x).correct()))
# Lemmatize wine reviews

from textblob import Word

wine_reviews['description'] = wine_reviews['description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

wine_reviews['description'].head()
# Sentiment Analysis 

def detect_polarity(text):

    return TextBlob(text).sentiment.polarity



wine_reviews['polarity'] = wine_reviews['description'].apply(detect_polarity)

wine_reviews.head(3)
max_pol = wine_reviews.polarity.max()

min_pol = wine_reviews.polarity.min()

print("Maximum polarity score is", max_pol, "while the minimum is", min_pol)
fig = plt.figure(figsize=(16,4))



sns.distplot(wine_reviews['polarity'], color='y')
fig = plt.figure(figsize=(16,6))

color = sns.cubehelix_palette(21, start=.5, rot=-.85)

sns.boxenplot(x='points', y='polarity', data=wine_reviews, palette=color)
wine_reviews['polarity'].corr(wine_reviews['points'])
len(wine_reviews['taster_name'].unique())
wine_reviews.sort_values(by='taster_name', inplace=True)

wine_reviews.groupby('taster_name').agg(['count'])
wine_reviews.loc[wine_reviews.taster_name == 'Christina Pickard']
# Roger Voss country data 

roger_voss = wine_reviews.loc[wine_reviews['taster_name']=='Roger Voss']

df = roger_voss.groupby(['country'],as_index=False).agg(['count']).reset_index()

df.columns = df.columns.droplevel(1)

df = df[['country', 'description']]

df.rename(columns={'description':'count'}, inplace=True)

df.sort_values('count', ascending=False, inplace=True)

df = df.reset_index(drop=True)

list_country = df['country'].tolist()

list_count = df['count'].tolist()
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



data = [go.Choropleth(

           locations = list_country,

           locationmode = 'country names',

           z = list_count,

           colorscale = 'Greens',

           autocolorscale = False,

           reversescale = True,

           marker = dict(line = dict(color = 'rgb(255,255,255)', width=1)),

           )]



layout = dict(title = 'Roger Voss Wine Reviews by Countries',

             geo = dict(scope = 'world'))

choromap = go.Figure(layout=layout, data=data)

iplot(choromap)
x = 14395/20172

y = 20153/20172

print("French wine reviews:" + " {:.2%}".format(x))  

print("European wine reviews:" + " {:.2%}".format(y))
fig = plt.figure(figsize=(16,3))

color = sns.color_palette('summer', 20)

sns.countplot(x='points', data=roger_voss, palette=color)
data = roger_voss.loc[(roger_voss.points >= 88) & (roger_voss.points <= 90)].sort_values(by='points')

color = sns.color_palette('spring', 3)

fig = plt.figure(figsize=(16,6))

sns.countplot(x='country', hue = 'points', data=data, palette=color)
# Getting winery review counts for 88 point wines 

wine_88 = data.loc[data.points==88]

wine_88 = wine_88.groupby(by='winery',as_index=False).count().sort_values(by='country',ascending=False).head(10).reset_index(drop=True)

list_winery = wine_88['winery'].tolist()

count_winery = wine_88['country'].tolist()

d = {'Winery': list_winery, 'Count': count_winery}

df = pd.DataFrame(d)

df['Points'] = '88'
# Getting winery review counts for 89 point wines 

wine_89 = data.loc[data.points==89]

wine_89 = wine_89.groupby(by='winery',as_index=False).count().sort_values(by='country',ascending=False).head(10).reset_index(drop=True)

wine_89

list_winery_two = wine_89['winery'].tolist()

count_winery_two = wine_89['country'].tolist()

d_two = {'Winery': list_winery_two, 'Count': count_winery_two}

d_two = pd.DataFrame(d_two)

d_two['Points'] = '89'

df = df.append(d_two)
# Getting winery review counts for 90 point wines 

wine_90 = data.loc[data.points==90]

wine_90.groupby(by='winery',as_index=False).count().sort_values(by='country',ascending=False).head(10).reset_index(drop=True)

list_winery_three = wine_89['winery'].tolist()

count_winery_three = wine_89['country'].tolist()

d_three = {'Winery': list_winery_three, 'Count': count_winery_three}

d_three = pd.DataFrame(d_three)

d_three['Points'] = '90'

df = df.append(d_three)
df.loc[df['Winery']=='DFJ Vinhos']
fig = plt.figure(figsize=(16,8))

sns.lineplot(x='Points', y='Count', hue='Winery', data=df)

plt.title('Reviews by Roger Voss')