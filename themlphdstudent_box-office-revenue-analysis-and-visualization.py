import pandas as pd

import numpy as np



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

plt.style.use('dark_background')



# display multiple output in single cell

from IPython.display import display

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



# data

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%time train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
train.head(n=10)
print("Shape of data is ")

train.shape

print('The total number of movies are',train.shape[0])
train.info()
train.describe(include='all')
# checking NULL value



train.isnull().sum()
train = train.drop(['belongs_to_collection', 'genres', 'crew', 'cast', 'Keywords', 

                  'spoken_languages', 'production_companies', 'production_countries', 'tagline','overview','homepage'], axis=1)
train['release_date'] = pd.to_datetime(train['release_date'], infer_datetime_format=True)

train['release_day'] = train['release_date'].apply(lambda t: t.day)

train['release_weekday'] = train['release_date'].apply(lambda t: t.weekday())

train['release_month'] = train['release_date'].apply(lambda t: t.month)



# Year was being interpreted as future dates in some cases so I had to adjust some values

train['release_year'] = train['release_date'].apply(lambda t: t.year if t.year < 2018 else t.year -100)
train[train['revenue'] == train['revenue'].max()]
train[['id','title','budget','revenue']].sort_values(['revenue'], ascending=False).head(10).style.background_gradient(subset='revenue', cmap='BuGn')
train[train['budget'] == train['budget'].max()]
train[['id','title','budget', 'revenue']].sort_values(['budget'], ascending=False).head(10).style.background_gradient(subset=['budget', 'revenue'], cmap='PuBu')
train[train['runtime'] == train['runtime'].max()]
plt.hist(train['runtime'].fillna(0) / 60, bins=40);

plt.title('Distribution of length of film in hours', fontsize=16, color='white');

plt.xlabel('Duration of Movie in Hours')

plt.ylabel('Number of Movies')
train[['id','title','runtime', 'budget', 'revenue']].sort_values(['runtime'],ascending=False).head(10).style.background_gradient(subset=['runtime','budget','revenue'], cmap='YlGn')
plt.figure(figsize=(20,12))

edgecolor=(0,0,0),

sns.countplot(train['release_year'].sort_values(), palette = "Dark2", edgecolor=(0,0,0))

plt.title("Movie Release count by Year",fontsize=20)

plt.xlabel('Release Year')

plt.ylabel('Number of Movies Release')

plt.xticks(fontsize=12,rotation=90)

plt.show()
train['release_year'].value_counts().head()
train[train['popularity']==train['popularity'].max()][['original_title','popularity','release_date','revenue']]
train[train['popularity']==train['popularity'].min()][['original_title','popularity','release_date','revenue']]
plt.figure(figsize=(20,12))

edgecolor=(0,0,0),

sns.distplot(train['popularity'], kde=False)

plt.title("Movie Popularity Count",fontsize=20)

plt.xlabel('Popularity')

plt.ylabel('Count')

plt.xticks(fontsize=12,rotation=90)

plt.show()
plt.figure(figsize=(20,12))

edgecolor=(0,0,0),

sns.countplot(train['release_month'].sort_values(), palette = "Dark2", edgecolor=(0,0,0))

plt.title("Movie Release count by Month",fontsize=20)

plt.xlabel('Release Month')

plt.ylabel('Number of Movies Release')

plt.xticks(fontsize=12)

plt.show()
train['release_month'].value_counts()
plt.figure(figsize=(20,12))

edgecolor=(0,0,0),

sns.countplot(train['release_day'].sort_values(), palette = "Dark2", edgecolor=(0,0,0))

plt.title("Movie Release count by Day of Month",fontsize=20)

plt.xlabel('Release Day')

plt.ylabel('Number of Movies Release')

plt.xticks(fontsize=12)

plt.show()
train['release_day'].value_counts()
plt.figure(figsize=(20,12))

sns.countplot(train['release_weekday'].sort_values(), palette='Dark2')

loc = np.array(range(len(train['release_weekday'].unique())))

day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.xlabel('Release Day of Week')

plt.ylabel('Number of Movies Release')

plt.xticks(loc, day_labels, fontsize=12)

plt.show()
train['release_weekday'].value_counts()