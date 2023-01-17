import numpy as np 

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt

import os

import seaborn as sns

import itertools

sns.set(style="darkgrid")
data = pd.read_csv("../input/IMDB-Movie-Data.csv")
print(data.columns)

print(50*'-')

print(data.dtypes)

print(50*'-')

print(data.shape)
data.rename({'Runtime (Minutes)': 'Duration', 'Revenue (Millions)':'Revenue'}, axis='columns',inplace=True)
print('The dataset contains NaN values: ',data.isnull().values.any())

print('Missing values in the dataset : ',data.isnull().values.sum())

for col in data.columns:

    nans=pd.value_counts(data[col].isnull())

    if len(nans)>1:

        print('Column: ',col,' , Missing values: ',nans[1])
Nans=data[pd.isnull(data).any(axis=1)]

print(Nans.head())
data.describe(include='all')
data.dropna(inplace=True)
actors=list(actor.split(',') for actor in data.Actors)

actors=list(itertools.chain.from_iterable(actors))

actors = [actor.strip(' ') for actor in actors]

actors_count=pd.value_counts(actors)

print('There are ',len(actors), 'different actors in the dataset,after removing NaN rows')
sns.pairplot(data,kind="reg")
fig, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(data.corr(), annot=True, fmt=".2f", linewidths=.5, ax=ax)

plt.show()
fig, axs = plt.subplots(2, 2, figsize=(25,15))

plt.suptitle('Boxplots of Duration,Rating,Votes and Revenue',fontsize=20)

sns.boxplot(data.Duration,ax=axs[0][0],color=sns.xkcd_rgb["cerulean"])

axs[0][0].set_xlabel('Duration (Minutes)',fontsize=14)

sns.boxplot(data.Rating,ax=axs[0][1],color='r')

axs[0][1].set_xlabel('Rating',fontsize=14)

sns.boxplot(data.Votes,ax=axs[1][0],color=sns.xkcd_rgb["teal green"])

axs[1][0].set_xlabel('Votes',fontsize=14)

sns.boxplot(data.Revenue,ax=axs[1][1],color=sns.xkcd_rgb["dusty purple"])

axs[1][1].set_xlabel('Revenue in millions',fontsize=14)

plt.show()
fig, axs = plt.subplots(2, 2, figsize=(25, 15))

plt.suptitle('Histograms of Duration,Rating,Votes and Revenue',fontsize=20)

sns.distplot(data.Duration,ax=axs[0][0],color=sns.xkcd_rgb["cerulean"])

axs[0][0].set_xlabel('Duration (Minutes)',fontsize=14)

sns.distplot(data.Rating,ax=axs[0][1],color='r')

axs[0][1].set_xlabel('Rating',fontsize=14)

sns.distplot(data.Votes,ax=axs[1][0],color=sns.xkcd_rgb["teal green"])

axs[1][0].set_xlabel('Votes',fontsize=14)

sns.distplot(data.Revenue,ax=axs[1][1],color=sns.xkcd_rgb["dusty purple"])

axs[1][1].set_xlabel('Revenue in millions',fontsize=14)

plt.show()
fig, axs = plt.subplots(2, 2, figsize=(25, 15))

plt.suptitle('Comparison of Rating and Metascore',fontsize=20)

sns.distplot(data.Metascore,ax=axs[0][1],color='g')

sns.distplot(data.Rating,ax=axs[0][0],color='r')

sns.boxplot(data.Metascore,ax=axs[1][1],color='g')

sns.boxplot(data.Rating,ax=axs[1][0],color='r')

plt.show()
plt.figure(figsize=(22,12))

plt.suptitle('Revenue by Metascore',fontsize=20)

sns.lineplot(data.Metascore,data.Revenue,color='g')

plt.xticks(size=14)

plt.yticks(size=14)

plt.show()
plt.figure(figsize=(22,10))

plt.title('Revenue by Rating',fontsize=20)

sns.lineplot(data.Rating,data.Revenue,color='r')

plt.xticks(size=14)

plt.yticks(size=14)

plt.show()
plt.figure(figsize=(25, 10))

plt.suptitle('Revenue by Rating, sized by Votes',fontsize=20)

sns.scatterplot(data.Rating,data.Revenue,hue=data.Votes,size=data.Votes,sizes=(50, 200))

plt.xlabel('Rating',fontsize=16)

plt.ylabel('Revenue',fontsize=16)

plt.xticks(size=14)

plt.yticks(size=14)

plt.show()
plt.figure(figsize=(25, 10))

plt.suptitle('Revenue by Metascore, sized by Votes',fontsize=20)

sns.scatterplot(data.Metascore,data.Revenue,hue=data.Votes,size=data.Votes,sizes=(50, 200))

plt.xlabel('Metascore',fontsize=16)

plt.ylabel('Revenue',fontsize=16)

plt.xticks(size=14)

plt.yticks(size=14)

plt.show()
plt.figure(figsize=(22,12))

plt.title('Most frequent Genre types',fontsize=20)

plt.xlabel('Genre')

sns.countplot(data.Genre,order=pd.value_counts(data.Genre).iloc[:15].index,palette=sns.color_palette("cubehelix", 15))

plt.xticks(size=16,rotation=30)

plt.yticks(size=16)

plt.show()
top_genre_groups=data.groupby(['Genre']).sum().sort_values(by='Revenue',ascending=False)[:10]

print('Total values for different Genre groups')

print('-'*60)

print(top_genre_groups[['Rank','Revenue','Metascore']])
plt.figure(figsize=(22,15))

plt.title('Genre groups with highiest total Revenue',fontsize=20)

sns.barplot(top_genre_groups.Revenue,top_genre_groups.index,palette=sns.color_palette("muted", 20))

plt.ylabel('Genre groups',fontsize=16)

plt.xlabel('Total Revenue',fontsize=16)

plt.xticks(size=14)

plt.yticks(size=16)

plt.show()
genres=list(genre.split(',') for genre in data.Genre)

genres=list(itertools.chain.from_iterable(genres))

genres=pd.value_counts(genres)

print('There are ',len(genres), 'different Genres in the dataset:')

print('-'*50)

print(genres)
plt.figure(figsize=(22,15))

plt.title('Most frequent Genre types',fontsize=20)

plt.xlabel('Count',fontsize=16)

sns.barplot(genres.values,genres.index,palette=sns.color_palette("muted", 20))

plt.xticks(size=16)

plt.yticks(size=16)

plt.show()
print('-'*30)

print('Top actors')

print('-'*30)

print(pd.value_counts(actors)[:10])
plt.figure(figsize=(22,15))

plt.title('Top actors',fontsize=20)

plt.ylabel('Actor',fontsize=16)

plt.xlabel('Number of movies participated',fontsize=16)

sns.barplot(actors_count.values,actors_count.index,order=actors_count[:20].index,palette=sns.color_palette("cubehelix", 25))

plt.xticks(size=16)

plt.yticks(size=16)

plt.show()
print('-'*80)

print('Top Directors in terms of total movies directed:')

print('-'*80)

print(pd.value_counts(data.Director)[:10])
plt.figure(figsize=(22,8))

plt.title('Top Directors (based on number of movies directed)',fontsize=20)

sns.countplot(data.Director,order=pd.value_counts(data.Director)[:20].index,palette=sns.color_palette("muted", 20))

plt.xlabel('Directors',fontsize=16)

plt.ylabel('Number of movies directed',fontsize=16)

plt.xticks(size=16,rotation=45)

plt.show()
top_directors=pd.DataFrame(columns=['director','mean_rating','mean_metascore'])

for director in data.Director.unique():

    #print(" %30s,   Mean Rating: %8.2f ,   Mean Metascore: %8.2f" % (director,data.Rating[data.Director==director].values.mean(),data.Metascore[data.Director==director].values.mean()))

    if len(data[data.Director==director].values)>=4:

        mean_rating=data.Rating[data.Director==director].values.mean()

        mean_metascore=data.Metascore[data.Director==director].values.mean()

        top_directors=top_directors.append( {'director': director ,

                           'mean_rating': mean_rating,

                           'mean_metascore':mean_metascore} , ignore_index=True)



top_directors=top_directors.sort_values(by=['mean_rating','mean_metascore'],ascending=False).reset_index(drop=True)

print(top_directors.head())
print('-'*80)

print('Best Directors based on mean Rating and Metascore (at least 4 movies directed)')

print('-'*80)

print(top_directors.head())

print('-'*80)

print('Worst Directors based on mean Rating and Metascore (at least 4 movies directed)')

print('-'*80)

print(top_directors.tail())

print('-'*80)
plt.figure(figsize=(25, 15))

plt.title('Top Directors based on mean Rating (At least 4 movies directed)',fontsize=20)

sns.barplot(top_directors.mean_rating[:10],top_directors.director[:10])

plt.xticks(size=14)

plt.yticks(size=16)

plt.xlabel('Mean Rating',fontsize=16)

plt.ylabel('Directors',fontsize=16)

plt.show()
plt.figure(figsize=(20, 12))

plt.title('Top Directors based on mean Metascore (At least 4 movies directed)',fontsize=20)

sns.barplot(top_directors.sort_values(['mean_metascore'],ascending=False).mean_metascore[:10],top_directors.sort_values(['mean_metascore'],ascending=False).director[:10])

plt.xticks(size=14)

plt.yticks(size=16)

plt.xlabel('Mean Metascore',fontsize=16)

plt.ylabel('Directors',fontsize=16)

plt.show()
top_actors=pd.DataFrame(columns=['actor','mean_rating','mean_metascore'])

for actor in pd.value_counts(actors).index:

    if len(data[data.Actors.str.contains(actor)].values)>=4:

        mean_rating=data.Rating[data.Actors.str.contains(actor)].values.mean()

        mean_metascore=data.Metascore[data.Actors.str.contains(actor)].values.mean()

        #print(actor,mean_rating,mean_metascore)

        top_actors=top_actors.append( {'actor': actor ,

                           'mean_rating': mean_rating,

                           'mean_metascore':mean_metascore} , ignore_index=True)



top_actors=top_actors.sort_values(by=['mean_rating','mean_metascore'],ascending=False).reset_index(drop=True)

print(top_actors[:10])
print('-'*80)

print('Best Actors based on Rating and Metascore, participated in at least 4 movies')

print('-'*80)

print(top_actors.head())

print('-'*80)

print('Worst Actors based on Rating and Metascore,participated in at least 4 movies')

print('-'*80)

print(top_actors.tail())

print('-'*80)
plt.figure(figsize=(20, 10))

plt.title('Top Actors based on mean Rating (participated in at least 4 movies)',fontsize=20)

sns.barplot(top_actors.mean_rating[:10],top_actors.actor[:10])

plt.xticks(size=16)

plt.yticks(size=16)

plt.ylabel('Actor',fontsize=16)

plt.ylabel('Mean Rating',fontsize=16)

plt.show()
plt.figure(figsize=(20, 10))

plt.title('Top Actors based on mean Metascore (participated in at least 4 movies)',fontsize=20)

sns.barplot(top_actors.sort_values(['mean_metascore'],ascending=False).mean_metascore[:10],top_actors.sort_values(['mean_metascore'],ascending=False).actor[:10])

plt.xticks(size=16)

plt.yticks(size=16)

plt.ylabel('Actor',fontsize=16)

plt.ylabel('Mean Metascore',fontsize=16)

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(25, 12))

plt.suptitle('Revenue and Ranking by Year',fontsize=20)

sns.scatterplot(data.Year,data.Revenue,ax=axs[0],color=sns.xkcd_rgb["dusty purple"])

sns.scatterplot(data.Year,data.Rank,ax=axs[1])

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(25, 12))

plt.suptitle('Rating and Metascore by Year',fontsize=20)

sns.scatterplot(data.Year,data.Rating,ax=axs[0],color='r')

sns.scatterplot(data.Year,data.Metascore,ax=axs[1],color='g')

plt.show()
top_revenues=data.sort_values('Revenue',ascending=False).reset_index(drop=True)[:10]
print('-'*100)

print('Best selling movies (Revenue in millions)')

print('-'*100)

print(top_revenues[['Title','Revenue']])
plt.figure(figsize=(15,12))

plt.title('Best selling movies',fontsize=20)

sns.barplot('Revenue','Title',data=top_revenues)

plt.ylabel('Movie Title',fontsize=16)

plt.xlabel('Revenue (millions)',fontsize=16)

plt.xticks(size=16,rotation=90)

plt.show()


best_rated_movies=data.sort_values('Rating',ascending=False).reset_index(drop=True)[:10]

print('-'*100)

print('Top rated movies')

print('-'*100)

print(best_rated_movies[['Title','Rating']])
plt.figure(figsize=(15,12))

plt.title('Top rated movies',fontsize=20)

sns.barplot('Rating','Title',data=best_rated_movies)

plt.ylabel('Movie Title',fontsize=16)

plt.xlabel('Rating',fontsize=16)

plt.xticks(size=16)

plt.yticks(size=16)

plt.show()
top_voted_movies=data.sort_values('Votes',ascending=False).reset_index(drop=True)[:10]

print('-'*100)

print('Top voted movies')

print('-'*100)

print(top_voted_movies[['Title','Votes']][:10])
plt.figure(figsize=(15,12))

plt.title('Most voted movies',fontsize=20)

sns.barplot('Votes','Title',data=top_voted_movies)

plt.ylabel('Movie Title',fontsize=16)

plt.xlabel('Votes',fontsize=16)

plt.xticks(size=16)

plt.yticks(size=16)

plt.show()
best_metascore=data.sort_values('Metascore',ascending=False).reset_index(drop=True)[:10]

plt.figure(figsize=(15,12))

sns.barplot('Metascore','Title',data=best_metascore)

plt.title('Movies with highiest Metascore',fontsize=20)

plt.ylabel('Movie Title',fontsize=16)

plt.xlabel('Metascore',fontsize=16)

plt.xticks(size=16)

plt.yticks(size=16)

plt.show()
print('-'*100)

print('Movies with highiest Metascore')

print('-'*100)

print(best_metascore[['Title','Metascore']])
plt.figure(figsize=(20, 10))

plt.title('Rank by Rating',fontsize=20)

sns.scatterplot('Rating','Rank',hue='Votes',size='Votes',data=data,sizes=(50, 200))

plt.ylabel('Rank',fontsize=16)

plt.xlabel('Rating',fontsize=16)

plt.xticks(size=16)

plt.yticks(size=16)

plt.show()

plt.figure(figsize=(20, 10))

plt.title('Rank by Metascore',fontsize=20)

sns.scatterplot('Metascore','Rank',hue='Votes',size='Votes',data=data,sizes=(50, 200))

plt.ylabel('Rank',fontsize=16)

plt.xlabel('Metascore',fontsize=16)

plt.xticks(size=16)

plt.yticks(size=16)

plt.show()
plt.figure(figsize=(20, 10))

plt.title('Rank by Revenue',fontsize=20)

sns.scatterplot('Revenue','Rank',hue='Votes',size='Votes',data=data,sizes=(50, 200))

plt.ylabel('Rank',fontsize=16)

plt.xlabel('Revenue',fontsize=16)

plt.xticks(size=16)

plt.yticks(size=16)

plt.show()
plt.figure(figsize=(20, 10))

plt.title('Rank by Votes',fontsize=20)

sns.scatterplot('Votes','Rank',hue='Revenue',size='Revenue',data=data,sizes=(50, 200))

plt.ylabel('Rank',fontsize=16)

plt.xlabel('Votes',fontsize=16)

plt.xticks(size=16)

plt.yticks(size=16)

plt.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

stopwords=set(STOPWORDS)

stopwords.update(['One','Two','Three','La'])
plt.subplots(figsize=(20,15))

plt.title('Movie Titles',fontsize=16)

text=str(data.Title)

wordcloud = WordCloud(stopwords=stopwords).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
plt.subplots(figsize=(20,15))

text=str(data.Description)

wordcloud = WordCloud( stopwords=stopwords).generate(text)

plt.title('Movie Descriptions',fontsize=16)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()