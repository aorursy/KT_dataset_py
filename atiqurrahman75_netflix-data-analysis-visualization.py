import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import operator

import json

import plotly.express as px
df = pd.read_csv('../input/netflix-titles/netflix_titles.csv')

df.head()
df.drop(['show_id', 'date_added', 'cast', 'description'], axis = 'columns', inplace = True)
df.isnull().sum()
a = df.isnull()

sns.heatmap(a, cmap='YlGnBu');
df['rating'].value_counts()
df['country'].value_counts()
df['country'].replace(np.nan, 'United States', inplace = True)

df['rating'].replace(np.nan, 'TV-MA', inplace= True)
plt.style.use('seaborn')



plt.pie(df['type'].value_counts(), colors= ['#3a9dbc', '#bec3de'], labels = ['Movie', 'TV Show'],  autopct='%1.1f%%',

        startangle=90);



plt.title('Percentage of Movies and TV Shows Available on Netflix')

plt.tight_layout();
rating_dict = df['rating'].value_counts().to_dict()
x = [i for i in rating_dict]

y =[ rating_dict[i] for i in rating_dict ]



plt.figure(figsize=(8,5), dpi=100)



plt.xlabel('Ratings')

plt.ylabel('Counts')

plt.title('Ratings of Shows available on Netflix')



plt.bar(x,y, color='#800000');



plt.tight_layout();
fig, ax = plt.subplots(figsize=(8,5), dpi=100)



bwid = 0.35



x = [i for i in rating_dict]

my = [i for i in df.loc[ df['type'] == 'Movie' ]['rating'].value_counts()]



ax.bar(np.arange(len(x)) - bwid/2, my, bwid, label='Movies');



tx = [ i for i in df.loc[ df['type'] == 'TV Show' ]['rating'].value_counts().sort_values(ascending=False).index.unique() ]

ty = [ i for i in df.loc[ df['type'] == 'TV Show' ]['rating'].value_counts()]



ax.bar(np.arange(len(tx)) + bwid/2, ty, bwid, label='TV Shows');



ax.set_title('Frequency of Movie and Tv Show Ratings in Netflix')

ax.set_xlabel('Ratings')

ax.set_ylabel('Counts')

ax.set_xticks(np.arange(len(x)))

ax.set_xticklabels(x);







ax.legend();

fig.tight_layout();
df['country'] = df['country'].str.split(', ')
uniq_countries = []



for i in df['country']: 

    for j in i:

        if j not in uniq_countries:

            uniq_countries.append(j)

            

total_countries = []



for i in df['country']:

    for j in i:

        total_countries.append(j)

        

country_dict = {}



for i in uniq_countries:

    c_dict = {}

    c_dict[i] = total_countries.count(i)

    country_dict.update(c_dict)

    

sorted_country_dict = dict( sorted(country_dict.items(), key=operator.itemgetter(1),reverse=True))
x = [i for i in sorted_country_dict][:10]

y = [country_dict[i] for i in sorted_country_dict ][:10]



fig, ax = plt.subplots(figsize=(8,5), dpi=100)





ax.set_xlabel('Countries')

ax.set_ylabel('Number of movies and tv-shows')

ax.set_title('Top 10 Countries with the highest number of Netflix Shows')



ax.barh(x,y, color='#800000', height=0.6);

ax.invert_yaxis();



fig.tight_layout();
x = [i for i in sorted(df['release_year'].unique()) ]

y = [i for i in df['release_year'].value_counts().sort_index() ]



plt.figure(figsize=(12,5), dpi=100)

plt.bar(x,y, color = 'r');



plt.title('Shows released on Netflix 1925-2020');

plt.xticks(x, rotation = 90, fontsize=5);

plt.xlabel('Year');

plt.ylabel('Count');



plt.tight_layout();



plt.show();
fig, ax = plt.subplots(2, 1, sharey = True, figsize = (10,6), dpi=100)



mx = [ i for i in sorted( df.loc[ df['type'].str.contains("Movie") ]['release_year'].unique() )[-20:] ]

my = [ i for i in df.loc[ df['type'].str.contains("Movie") ]['release_year'].value_counts().sort_index()[-20:] ]



ax[0].bar(mx, my, color='#884EA0');

ax[0].set_title('Frequency of Movies released on Netflix')

ax[0].set_xticks(mx)



tx = [ i for i in sorted( df.loc[ df['type'].str.contains("TV Show") ]['release_year'].unique() )[-20:] ]

ty = [ i for i in df.loc[ df['type'].str.contains("TV Show") ]['release_year'].value_counts().sort_index()[-20:] ]



ax[1].bar(tx, ty, color='#148F77');

ax[1].set_title('Frequency of TV Shows released on Netflix')

ax[1].set_xticks(tx)



ax[1].set_xlabel('Year')

ax[1].set_ylabel('Frequency')



fig.tight_layout();
# oldest shows available netflix

old_shows_df = df.sort_values('release_year')

old_shows_df = old_shows_df[['title', 'type', 'release_year']][:20]

old_shows_df
# documentaries availabele in netflix

documentary_df = df[ df['listed_in'].str.contains('Documentaries') ]

documentary_df = documentary_df[[ 'title', 'director','release_year' ]]

documentary_df.head()
# kids shows available on netflix

kidshow_df = df[ df['listed_in'].str.contains("Kids' TV") ]

kidshow_df = kidshow_df[[ 'title','release_year' ]]

kidshow_df.head()
df['listed_in'] = df['listed_in'].str.split(', ')
genre_list = []



for i in df['listed_in']: 

    for j in i:

        if j not in genre_list:

            genre_list.append(j)

  



all_genre = []



for i in df['listed_in']:

    for j in i:

        all_genre.append(j)   

        

        

genre_dict = {}



for i in genre_list:

    g_dict = {}

    g_dict[i]  = all_genre.count(i)

    genre_dict.update(g_dict)

    

sorted_genre_dict = dict( sorted(genre_dict.items(), key=operator.itemgetter(1), reverse=True))
fig, ax = plt.subplots(figsize=(8,5), dpi=100)



x = [ i for i in sorted_genre_dict ][:20]

y = [ genre_dict[i] for i in sorted_genre_dict ][:20]



ax.barh(x, y, color='#9E14AB');



ax.set_title('Top 20 Genres of Shows available on Netflix')



ax.invert_yaxis();

fig.tight_layout();
m_all_genre = []



for i in df.loc[ df['type'] == 'Movie' ]['listed_in']:

    for j in i:

        m_all_genre.append(j)

        

t_all_genre = []



for i in df.loc[ df['type'] == 'TV Show' ]['listed_in']:

    for j in i:

        t_all_genre.append(j)



mgenre_dict = {}



for i in genre_list:

    mg_dict = {}

    mg_dict[i]  = m_all_genre.count(i)

    mgenre_dict.update(mg_dict)

    

tgenre_dict = {}



for i in genre_list:

    tg_dict = {}

    tg_dict[i]  = t_all_genre.count(i)

    tgenre_dict.update(tg_dict)



sorted_mgenre_dict = dict( sorted(mgenre_dict.items(), key=operator.itemgetter(1), reverse=True))

sorted_tgenre_dict = dict( sorted(tgenre_dict.items(), key=operator.itemgetter(1), reverse=True))
fig, ax = plt.subplots(1, 2, sharex=True, figsize=(12,4), dpi = 150)



mx = [ i for i in sorted_mgenre_dict ][:10]

my = [ mgenre_dict[i] for i in sorted_mgenre_dict ][:10]



ax.flat[0].barh(mx, my, height=0.5, color='#138D75');

ax.flat[0].set_title('Top 10 Genres of Movies available on Netflix')



tx = [ i for i in sorted_tgenre_dict ][:10]

ty = [ tgenre_dict[i] for i in sorted_tgenre_dict ][:10]



ax.flat[1].barh(tx, ty, height=0.5, color='#CB4335');

ax.flat[1].set_title('Top 10 Genres of TV Shows available on Netflix')



ax[0].invert_yaxis();

ax[1].invert_yaxis();

fig.tight_layout();
world_map = json.load(open('/kaggle/input/worldcountries-geojson/worldcountries_geojson.geojson', 'r'))
world_map['features'][1].keys()
country_id_map = {}



for feature in world_map['features']:

    feature['id'] = feature['properties']['ISO_A3']

    country_id_map[feature['properties']['ADMIN']] = feature['id']

    

a = {

    'United States' : 'USA',

    'Hong Kong' : 'HKG',

    'Serbia' : 'SRB',

    'West Germany': 'DEU',

    'Soviet Union' : 'RUS',

    'United States,' : 'VIR',

    'Poland,' : 'POL',

    'United Kingdom,' : 'DJ',

    'Vatican City' : 'VAC',

    'East Germany' : 'EDEU,'

              }

country_id_map.update(a)
country_df = pd.DataFrame(list(sorted_country_dict.items()), columns = ['Country', 'Counts'])

country_df['id'] = country_df['Country'].apply(lambda x: country_id_map[x])
import plotly.io as pio

pio.renderers.default = 'notebook' 
fig = px.choropleth_mapbox(country_df, locations='id', geojson=world_map, color = 'Counts',

                   hover_name='Country',

                          mapbox_style='carto-positron',

                           center={'lat': 30, 'lon': 0},

                          zoom=1,

                          opacity=0.7 )



fig.update_geos(fitbounds='locations')

pio.show(fig);