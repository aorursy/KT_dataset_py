import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

%matplotlib inline
netflix = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
netflix.head()
netflix.info()
netflix.isnull().sum()
netflix['cast'].nunique()
netflix = netflix.drop(['show_id','director','cast'], axis=1)
netflix.isnull().sum()/len(netflix)*100
netflix = netflix.dropna()
sns.heatmap(netflix.isnull(), vmin=0, vmax=1)
netflix['date_added']
netflix['month_added'] = netflix['date_added'].apply(lambda x : x.lstrip().split(' ')[0])

netflix['year_added'] = netflix['date_added'].apply(lambda x : x.lstrip().split(' ')[-1])
movie_added = netflix[netflix['type']=='Movie'].groupby(['year_added','month_added']).count()['title'].reset_index()

tv_show_added = netflix[netflix['type']=='TV Show'].groupby(['year_added','month_added']).count()['title'].reset_index()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
movie_added.pivot("month_added", "year_added", "title").reindex(index = months)
data = movie_added.pivot("month_added", "year_added", "title").reindex(index = months)



fig = plt.figure(figsize = (8,10))

gs = fig.add_gridspec(4, 1)

ax1 =  fig.add_subplot(gs[0:3,:])

ax2 =  fig.add_subplot(gs[3,:])



sns.heatmap(data, cmap =sns.light_palette("red"), annot=True, fmt ='.0f', ax = ax1, cbar=False)

bottom, top = ax1.get_ylim()

ax1.set_ylim(bottom + 0.5, top - 0.5)



pal = sns.light_palette("red",n_colors =len(data.columns), reverse = True)

rank = data.sum().argsort().argsort()

sns.barplot(data = data, estimator = sum, dodge = False, ax = ax2, palette = np.array(pal[::-1])[rank])
data = tv_show_added.pivot("month_added", "year_added", "title").reindex(index = months)



fig = plt.figure(figsize = (6,10))

gs = fig.add_gridspec(4, 1)

ax1 =  fig.add_subplot(gs[0:3,:])

ax2 =  fig.add_subplot(gs[3,:])



sns.heatmap(data, cmap =sns.light_palette("orange"), annot = True, fmt ='.0f', ax = ax1, cbar=False)

bottom, top = ax1.get_ylim()

ax1.set_ylim(bottom + 0.5, top - 0.5)



pal = sns.light_palette("orange",n_colors =len(data.columns), reverse = True)

rank = data.sum().argsort().argsort()

sns.barplot(data = data,  estimator = sum, dodge = False, ax = ax2, palette=np.array(pal[::-1])[rank])
netflix.groupby('listed_in').count()['title']
title = []

genre = []

types = []



def title_genre (df):

    for i in range(len(df['listed_in'].split(', '))):

        genre.append(df['listed_in'].split(', ')[i])

        title.append(df['title'])

        types.append(df['type'])
netflix.apply(title_genre, axis=1)
genre_table = pd.DataFrame(data = {'Titles':title, 'Genre':genre, 'Type':types})

genre_table
genre_table = genre_table.groupby(['Type','Genre']).count().reset_index()

movie_genre = genre_table[genre_table['Type'] == 'Movie'].sort_values('Titles', ascending = False)

tv_show_genre = genre_table[genre_table['Type'] == 'TV Show'].sort_values('Titles', ascending = False)
fig = px.pie(movie_genre, values='Titles', 

             names='Genre', 

             title="Movie Genre", 

             width = 750)



fig.update_traces(textposition='inside', 

                  textinfo='percent+label')



fig.update_layout(uniformtext_minsize=8, 

                  uniformtext_mode='hide')



fig.show()
fig = px.pie(tv_show_genre, 

             values='Titles', 

             names='Genre', 

             title="TV Show Genre", 

             width = 800)



fig.update_traces(textposition='inside', 

                  textinfo='percent+label')



fig.update_layout(uniformtext_minsize=8, 

                  uniformtext_mode='hide')



fig.show()
rating = netflix.groupby(['type','rating']).count()['title'].reset_index()
rating
fig = px.sunburst(rating, 

                  path =['type','rating'], 

                  values='title', 

                  width = 600, 

                  title = "Rating")



fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
netflix.groupby('country').count()['title']
title = []

country = []

types = []



def title_country (df):

    for i in range(len(df['country'].split(', '))):

        country.append(df['country'].split(', ')[i])

        title.append(df['title'])

        types.append(df['type'])
netflix.apply(title_country, axis=1)
country_table = pd.DataFrame(data = {'Titles':title, 'Country':country, 'Type':types})

country_table
country_counted = country_table.groupby(['Type','Country']).count().reset_index()

country_counted
country_counted = country_counted.pivot(index = 'Country', columns='Type', values ='Titles').reset_index()

country_counted
country_counted = country_counted.fillna(value = 0)

country_counted['Total']= country_counted['Movie']+country_counted['TV Show']

country_counted
Top_22_country = country_counted.sort_values('Total', ascending = False).head(22)

Top_22_country
f, ax = plt.subplots(figsize = (14,8))



sns.barplot(x="Total", y='Country',

            data=Top_22_country,

            label="TV Show", 

            color="g")



sns.barplot(x="Movie", y='Country',

            data=Top_22_country,

            label="Movie", 

            color="Turquoise")



ax.legend(ncol=2, loc="lower right", frameon=True)



ax.set(ylabel="",

       xlabel="", 

       title = "Top 22 Country")

sns.despine(left=True, bottom=True)
netflix['time'] = netflix['duration'].apply(lambda x : int(x.lstrip().split(' ')[0]))

netflix['unit'] = netflix['duration'].apply(lambda x : x.lstrip().split(' ')[-1])
fig = px.histogram(netflix[netflix['type']=='Movie'], 

                   x="time", marginal="box", 

                   nbins=40, 

                   color_discrete_sequence=['SlateGrey'], 

                   title= "Movie's Duration")

fig.show()
netflix[netflix['duration'] == '312 min']
f, ax = plt.subplots(figsize=(8,6))

sns.countplot(x='time', data = netflix[netflix['type']=='TV Show'], palette ='pastel')

ax.set(xlabel = 'Seasons',title="TV Show's Seasons")
netflix[netflix['duration'] == '15 Seasons']