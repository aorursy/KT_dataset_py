import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline



SEED = 0

DATA_DIR = '../input/criterion-movies-collection'

FILENAME = 'data.csv'



DATA_FILE = f'{DATA_DIR}/{FILENAME}'
data = pd.read_csv(DATA_FILE)

data.drop(data.columns[0], axis=1, inplace=True)
data.isnull().sum()
data.dropna(subset=['Year', 'Country'], inplace=True)

data.drop('Image', axis=1, inplace=True)



data['Year'] = data['Year'].astype('int32')
print(data.shape)

data.sample(10)
# top italian directors

italian_movies = data.loc[data['Country'] == 'Italy']

it_movies_by_director = italian_movies.groupby('Director').agg({ 'Title': 'count' })

it_movies_sorted = it_movies_by_director.sort_values(by='Title', ascending=False).rename(columns={'Title': 'Number of Movies'})

it_movies_sorted.head(8)
descending_order = data['Director'].value_counts().sort_values(ascending=False).iloc[:10].index

plt.rcParams['figure.figsize']=(20,6)

ax = sns.countplot(x='Director', data=data, palette = 'Spectral', order=descending_order)

ax.set_title(label='Number of films by Director', fontsize=20)

plt.show()
import matplotlib.ticker as ticker



descending_order = data['Country'].value_counts().sort_values(ascending=False).iloc[:10].index

ncount = len(data)



plt.figure(figsize=(20,6))

ax = sns.countplot(x="Country", data=data, order=descending_order, palette = 'Spectral')

plt.title('Number of films by Country', fontsize=20)



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom', fontsize=12)

plt.show()
data['Decade'] = data['Year'].apply(lambda x: ((x // 10) % 10) * 10)
descending_order = data['Decade'].value_counts().sort_values(ascending=False).iloc[:10].index

ncount = len(data)



plt.figure(figsize=(20,6))

ax = sns.countplot(x='Decade', data=data, order=descending_order, palette = 'Spectral')

plt.title('Number of films by Decade', fontsize=20)



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom', fontsize=12)

plt.show()
plt.figure(figsize = (12, 12))

text = ' '.join(data['Description'].fillna('').values)

wordcloud = WordCloud(margin=10, background_color='white', colormap='PuOr', width=1200, height=1000).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Top words used in Description', fontsize=20)

plt.axis('off')

plt.show()
plt.figure(figsize = (12, 12))

text = ' '.join(data['Title'].fillna('').values)

wordcloud = WordCloud(margin=10, background_color='white', colormap='coolwarm', width=1200, height=1000).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Top words used in Title', fontsize=20)

plt.axis('off')

plt.show()