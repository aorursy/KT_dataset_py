import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import warnings

warnings.filterwarnings('ignore')



# Allow several prints in one cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

file = '/kaggle/input/netflix-shows/netflix_titles.csv'

df = pd.read_csv(file)

df.head()
df.shape

# df.date_added.dtype

df.describe()
t = '''show_id:Unique ID for every Movie / Tv Show

type:Identifier - A Movie or TV Show

title:Title of the Movie / Tv Show

director:Director of the Movie

cast:Actors involved in the movie / show

country:Country where the movie / show was produced

date_added:Date it was added on Netflix

release_year:Actual Release year of the move / show

rating:TV Rating of the movie / show

duration:Total Duration - in minutes or number of seasons

listed_in:Genere

description:The summary description'''

cols = [r.split(':')[0] for r in t.split('\n')]

des = [r.split(':')[1] for r in t.split('\n')]



data_info = pd.DataFrame({'column_name': cols, 'description': des})

data_info
df["date_added"] = pd.to_datetime(df['date_added'])

df['year_added'] = df['date_added'].dt.year

df['month_added'] = df['date_added'].dt.month

# df['add_year'] = df['date_added'].fillna('Null').apply(lambda x: 2020 if x == 'Null' else int(x.split(', ')[-1]))



df['seaons'] = df.apply(lambda x: int(x.duration.split()[0]) if "Season" in x.duration else np.nan, axis=1)

df['length'] = df.apply(lambda x: np.nan if "Season" in x.duration else int(x.duration.split()[0]), axis=1)

df[df['year_added'] == 2020].head()
movies = df[df['type'] == 'Movie']

tv_shows = df[df['type'] == 'TV Show']

print("there are: \n{} movies and \n{} TV shows in the \n{} records".format(movies.show_id.count().sum(), tv_shows.shape[0], len(df)))
per_year = df.groupby(['year_added', 'type'])['show_id'].count().reset_index() #unstack().unstack()



f,a = plt.subplots(1,1,figsize=(14,8))

g = sns.barplot(x='year_added',

            y='show_id',

            hue='type',

            data=per_year,

            palette='Blues_d',

            ax = a

            )

a.set_ylabel('Counts')

a.set_xlabel('')

plt.xticks(rotation='30');
m_directors, tv_directors = [], []

for c in movies.director.fillna(""):

    m_directors.extend([ci.strip() for ci in c.split(',')])

for c in tv_shows.director.fillna(""):

    tv_directors.extend([ci.strip() for ci in c.split(',')])    

m_directors = pd.Series(m_directors)

# m_casts = m_casts[m_casts != ""]

tv_directors = pd.Series(tv_directors)

m_directors.value_counts().head()

tv_directors.value_counts().head()
f,a = plt.subplots(1,2,figsize=(18,6))

f.subplots_adjust(wspace = .4)

sns.barplot(m_directors.value_counts()[1:16], m_directors.value_counts()[1:16].index, ax=a[0])

a[0].set_title('The 15 most productive movie directors', fontsize=15, fontweight='bold')

a[0].set_xlabel('')

sns.barplot(tv_directors.value_counts()[1:16], tv_directors.value_counts()[1:16].index, ax=a[1])

a[1].set_title('The 15 most productive TV show directors', fontsize=15, fontweight='bold')

a[1].set_xlabel('');
m_casts, tv_casts = [], []

for c in movies.cast.fillna(""):

    m_casts.extend([ci.strip() for ci in c.split(',')])

for c in tv_shows.cast.fillna(""):

    tv_casts.extend([ci.strip() for ci in c.split(',')])    

m_casts = pd.Series(m_casts)

# m_casts = m_casts[m_casts != ""]

tv_casts = pd.Series(tv_casts)

m_casts.value_counts().head()

tv_casts.value_counts().head()
f,a = plt.subplots(1,2,figsize=(18,6))

f.subplots_adjust(wspace = .4)

sns.barplot(m_casts.value_counts()[1:16], m_casts.value_counts()[1:16].index, ax=a[0])

a[0].set_title('The 15 most popular movie stars', fontsize=15, fontweight='bold')

a[0].set_xlabel('')

sns.barplot(tv_casts.value_counts()[1:16], tv_casts.value_counts()[1:16].index, ax=a[1])

a[1].set_title('The 15 most popular TV show stars', fontsize=15, fontweight='bold')

a[1].set_xlabel('');
people = set(m_directors.value_counts()[1:16].index.tolist() +

             tv_directors.value_counts()[1:16].index.tolist() + 

             m_casts.value_counts()[1:16].index.tolist() + 

             tv_casts.value_counts()[1:16].index.tolist()

            )

print(people)
networks,works = {}, {}



for i,r in df.fillna('').iterrows():

    group = [p.strip() for p in r['director'].split(',') if p != '']

    group.extend([p.strip() for p in r['cast'].split(',') if p != ''])

#     print(group)

    for p in group:

        if len(p) != 0 and p in people:

            c = networks.get(p, set())

            networks[p] = c.union(set(group))

            works[p] = works.get(p, 0) + 1

        

#     break

for p in networks:

    networks[p] = len(networks.get(p)) - 1

# pd.concat([pd.Series(networks), pd.Series(works)], axis = 1)

network_vs_work = pd.DataFrame({'network_size':pd.Series(networks), 'works':pd.Series(works)})
ax = sns.regplot(x='network_size', y='works', data=network_vs_work, 

           fit_reg=False,y_jitter=0, scatter_kws={'alpha':0.2});

#            size = 6, aspect =2);

for i, r in network_vs_work.iterrows():

    ax.text(r.network_size+0.2, r.works, i)

ax.figure.set_size_inches(18, 10);
network_vs_work = network_vs_work.reset_index().rename(columns={'index': 'person'})
network_vs_work['movie_or_tv'] = network_vs_work.apply(lambda x: 'Movie' if x['person'] in set(m_directors.value_counts()[1:16].index.tolist() + 

              m_casts.value_counts()[1:16].index.tolist()) else 'TV', axis=1)

network_vs_work.sample(6)
network_vs_work['dir_or_star'] = network_vs_work.apply(lambda x: 'Director' if x['person'] in set(m_directors.value_counts()[1:16].index.tolist() + 

              tv_directors.value_counts()[1:16].index.tolist()) else 'Star', axis=1)

network_vs_work.sample(6)
ax = sns.scatterplot(x='network_size', y='works', hue='movie_or_tv', data=network_vs_work);

#            size = 6, aspect =2);

for i, r in network_vs_work.iterrows():

    ax.text(r.network_size+0.2, r.works, r.person)

ax.figure.set_size_inches(18, 10);
ax = sns.scatterplot(x='network_size', y='works', hue='dir_or_star', data=network_vs_work);

#            size = 6, aspect =2);

for i, r in network_vs_work.iterrows():

    ax.text(r.network_size+0.2, r.works, r.person)

ax.figure.set_size_inches(18, 10);
df[df['cast'].fillna('').str.contains('David Attenborough')].sort_values(by='year_added')
movies.country.nunique()

movies.country.value_counts().sum()

movies.country.value_counts()[:20].sum()

movies.country.value_counts()[:20].sum() / movies.country.value_counts().sum() * 100



movies.country.value_counts().head(20)
f,a = plt.subplots(1,2,figsize=(18,6))

f.subplots_adjust(wspace = .4)

sns.barplot(movies.country.value_counts().head(20), movies.country.value_counts().head(20).index, ax=a[0])

a[0].set_title('The 20 most productive (Movies) countries', fontsize=15, fontweight='bold')

a[0].set_xlabel('')

sns.barplot(tv_shows.country.value_counts().head(20), tv_shows.country.value_counts().head(20).index, ax=a[1])

a[1].set_title('The 20 most productive (TV Shows) countries', fontsize=15, fontweight='bold')

a[1].set_xlabel('');
f,a = plt.subplots(1,1,figsize=(18,6))

# f.subplots_adjust(wspace = .4)

sns.barplot(df.rating.value_counts().index, df.rating.value_counts(), ax=a)

a.set_title('Works per category', fontsize=15, fontweight='bold')

a.set_ylabel('Counts')

a.set_xlabel('Rating');
# source: https://www.kaggle.com/vikassingh1996/netflix-movies-and-shows-plotly-recommender-sys

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  height = 1000, max_words = 121).generate(' '.join(df['title']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in Title',fontsize = 30)

plt.show();
'''Plotly visualization .'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook



def pie_plot(cnt_srs, title):

    labels=cnt_srs.index

    values=cnt_srs.values

    trace = go.Pie(labels=labels, 

                   values=values, 

                   title=title, 

                   hoverinfo='percent+value', 

                   textinfo='percent',

                   textposition='inside',

                   hole=0.7,

                   showlegend=True,

                   marker=dict(colors=plt.cm.viridis_r(np.linspace(0, 1, 14)),

                               line=dict(color='#000000',

                                         width=2),

                              )

                  )

    return trace



py.iplot([pie_plot(df['rating'].value_counts(), 'Content Type')]);
# source: https://www.kaggle.com/subinium/storytelling-with-data-netflix-ver



import plotly.express as px

year_country2 = df.groupby('year_added')['country'].value_counts().reset_index(name='counts')



fig = px.choropleth(year_country2, locations="country", color="counts", 

                    locationmode='country names',

                    animation_frame='year_added',

                    range_color=[0,200],

                    color_continuous_scale=px.colors.sequential.OrRd

                   )



fig.update_layout(title='Comparison by country')

fig.show();