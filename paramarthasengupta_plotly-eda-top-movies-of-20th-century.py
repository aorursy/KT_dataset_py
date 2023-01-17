# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
movie=pd.read_csv('/kaggle/input/top-movies-database-19202000s/Movie Dataset.csv',encoding='latin1')

movie.head()
movie.isnull().sum()
movie.dtypes
yr_cnt=movie.groupby('Year').apply(lambda x:x['Title'].count()).reset_index(name='Count')

fig = px.bar(yr_cnt, y='Count', x='Year', text='Count',title='Number of Succesful Movies launched by year of the 20th Century')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
yr_cnt=movie.groupby(['Year','Subject']).apply(lambda x:x['Title'].count()).reset_index(name='Count')

plt.figure(figsize=(12,8))

fig = px.area(yr_cnt, x="Year", y="Count",color="Subject",title='Count of Movies Genres by the Years')

fig.show()
movie_len=movie[movie['Length'].isnull()==False]

movie_length=movie_len.groupby('Year').apply(lambda x:np.average(x['Length'])).reset_index(name='Average_Length')

plt.figure(figsize=(8,8))

sns.regplot(movie_length['Year'],movie_length['Average_Length'],color='red',logx=True)

plt.xlabel('Years',size=15)

plt.ylabel('Average Movie Length',size=15)

plt.title('Variation of Movie Length over the Years',size=20)

yr_sub_avg=movie.groupby(['Year','Subject']).apply(lambda x:x['Length'].mean()).reset_index(name='Average Movie Time')

plt.figure(figsize=(12,8))

fig = px.area(yr_sub_avg, x="Year", y="Average Movie Time",color='Subject')

fig.show()
movie_check=movie[movie['Popularity'].isnull()==False]

movie_len_pop=movie_check[['Length','Year','Popularity','Title','Director','Actor','Actress']]

fig = px.scatter(movie_len_pop, x='Year', y='Length',

                size='Popularity',color='Popularity',

                 hover_data=['Title','Director','Actor','Actress'],

                 title='Variation of Movie Length over the years, and estimating its populartiy')

fig.show()
movie_pop=movie[['Title','Year','Popularity','Subject']]

movie_pop.sort_values(by='Popularity',ascending=False,inplace=True)

movie_top=movie_pop[['Year','Title','Subject']][:20]

print('The Top 20 most popular movies are:\n',movie_top)
word_list=movie.Title.tolist()

strr=''

for i in word_list:

    strr=strr+i+' '

wordcloud = WordCloud(width = 1200, height = 1200, 

                background_color ='white',collocations=False,  

                min_font_size = 12).generate(strr) 

plt.figure(figsize = (8, 8), facecolor = 'grey') 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)

plt.title('WordCloud using the Movie Names',size=40)
actor_cnt=movie.groupby('Actor').apply(lambda x:x['Title'].count()).reset_index(name='Movie Counts')

actor_cnt.sort_values(by='Movie Counts',ascending=False,inplace=True)

top_actor=actor_cnt[:10]

fig = px.bar(top_actor, x='Actor', y='Movie Counts',

             color='Movie Counts',title='Top 10 Actors who have acted in most number of Movies ')

fig.show()

movie_actor=movie[movie['Actor'].isnull()==False]

word_list_2=movie_actor.Actor.tolist()

strr=''

for i in word_list_2:

    strr=strr+i+' '

wordcloud = WordCloud(width = 1200, height = 1200, 

                background_color ='white',collocations=False,  

                min_font_size = 12).generate(strr) 

plt.figure(figsize = (8, 8), facecolor = 'cyan') 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)

plt.title('WordCloud using the Movie Actor names',size=40)
actress_cnt=movie.groupby('Actress').apply(lambda x:x['Title'].count()).reset_index(name='Movie Counts')

actress_cnt.sort_values(by='Movie Counts',ascending=False,inplace=True)

top_actress=actress_cnt[:10]

fig = px.bar(top_actress, x='Actress', y='Movie Counts',

             color='Movie Counts',title='Top 10 Actresses who have acted in most number of Movies ')

fig.show()

movie_actress=movie[movie['Actress'].isnull()==False]

word_list_2=movie_actress.Actress.tolist()

strr=''

for i in word_list_2:

    strr=strr+i+' '

wordcloud = WordCloud(width = 1200, height = 1200, 

                background_color ='white',collocations=False,  

                min_font_size = 12).generate(strr) 

plt.figure(figsize = (8, 8), facecolor = 'pink') 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)

plt.title('WordCloud using the Movie Actress names',size=40)
director_cnt=movie.groupby('Director').apply(lambda x:x['Title'].count()).reset_index(name='Movie Counts')

director_cnt.sort_values(by='Movie Counts',ascending=False,inplace=True)

top_director=director_cnt[:10]

fig = px.bar(top_director, x='Director', y='Movie Counts',

             color='Movie Counts',title='Top 10 Directors who have directed most number of Movies ')

fig.show()

movie_director=movie[movie['Director'].isnull()==False]

word_list_2=movie_director.Director.tolist()

strr=''

for i in word_list_2:

    strr=strr+i+' '

wordcloud = WordCloud(width = 1200, height = 1200, 

                background_color ='black',collocations=False,  

                min_font_size = 12).generate(strr) 

plt.figure(figsize = (8, 8), facecolor = 'grey') 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)

plt.title('WordCloud using the Movie Director names',size=40)
act_cnt=movie.groupby(['Actor','Actress']).apply(lambda x:x['Title'].count()).reset_index(name='Movie Counts')

act_cnt.sort_values(by='Movie Counts',ascending=False,inplace=True)

print('The Top 10 Actress of the 20th Century were:\n',act_cnt[:10])
act_pop=movie.groupby(['Actor','Actress']).apply(lambda x:x['Popularity'].mean()).reset_index(name='Average Popularity')

act_overall=pd.merge(act_cnt,act_pop,how='inner',left_on=['Actor','Actress'],right_on=['Actor','Actress'])

fig = px.scatter(act_overall, x="Movie Counts", y="Average Popularity",

                 size="Movie Counts",color="Average Popularity",

                 hover_data=['Actor','Actress'],

                 title='Popularity vs Movie count metrics for the most succesful Hollywood pairs of the 20th Century')

fig.show()
cast_cnt=movie.groupby(['Director','Actor','Actress']).apply(lambda x:x['Title'].count()).reset_index(name='Movie Counts')

cast_cnt.sort_values(by='Movie Counts',ascending=False,inplace=True)

cast_pop=movie.groupby(['Director','Actor','Actress']).apply(lambda x:x['Popularity'].mean()).reset_index(name='Average Popularity')

cast_overall=pd.merge(cast_cnt,cast_pop,how='inner',left_on=['Director','Actor','Actress'],right_on=['Director','Actor','Actress'])

fig = px.scatter(cast_overall, x="Movie Counts", y="Average Popularity",

                size="Movie Counts",color="Average Popularity",

                 hover_data=['Director','Actor','Actress'],

                 title='Popularity vs Movie count metrics for the most succesful Hollywood Cast (Director and Actors)')

fig.show()
actor_genre=movie.groupby('Actor').apply(lambda x:x['Subject'].nunique()).reset_index(name='# Genres')

actress_genre=movie.groupby('Actress').apply(lambda x:x['Subject'].nunique()).reset_index(name='# Genres')

actor_genre.sort_values(by='# Genres',ascending=False,inplace=True)

actress_genre.sort_values(by='# Genres',ascending=False,inplace=True)

top_actor_genre=actor_genre[:10]

top_actor_genre['Gender']='Male'

top_actress_genre=actress_genre[0:10]

top_actress_genre['Gender']='Female'

top_actor_genre.columns=['Performer',"# Genres",'Gender']

top_actress_genre.columns=['Performer',"# Genres",'Gender']

top_performer_genre=pd.concat([top_actor_genre,top_actress_genre])

fig = px.scatter(top_performer_genre, x='Performer', y='# Genres',

                size='# Genres',color='Gender',

                 hover_data=['Performer'],

                 title='Top Performers of the 20th century who have worked in multiple Genres')

fig.show()
movie_filter=movie[movie['Title'].isnull()==False]

movie_filter=movie_filter[movie_filter['Actor'].isin(top_actor_genre.Performer.tolist())]

movie_filter=movie_filter[movie_filter['Subject'].isnull()==False]

actor_genre=movie_filter.groupby(['Actor','Subject']).apply(lambda x:x['Title'].count()).reset_index(name='# Genres wise Movies')

plt.figure(figsize=(8,10))

pivot_actor=pd.pivot(actor_genre,index='Actor',columns='Subject',values='# Genres wise Movies')

ax=sns.heatmap(pivot_actor,annot=True,fmt='g',cmap='Spectral_r')

plt.xlabel('Genres',size=20)

plt.ylabel('Actors',size=25)

plt.title('Genre wise filmography of the most verstaile Actors',size=20)

movie_filter_2=movie[movie['Title'].isnull()==False]

movie_filter_2=movie_filter_2[movie_filter_2['Actress'].isin(top_actress_genre.Performer.tolist())]

movie_filter_2=movie_filter_2[movie_filter_2['Subject'].isnull()==False]

actress_genre=movie_filter_2.groupby(['Actress','Subject']).apply(lambda x:x['Title'].count()).reset_index(name='# Genres wise Movies')

plt.figure(figsize=(8,10))

pivot_actress=pd.pivot(actress_genre,index='Actress',columns='Subject',values='# Genres wise Movies')

ax=sns.heatmap(pivot_actress,annot=True,fmt='g',cmap='copper_r')

plt.xlabel('Genres',size=20)

plt.ylabel('Actresses',size=25)

plt.title('Genre wise filmography of the most verstaile Actresses',size=20)
director_genre=movie.groupby('Director').apply(lambda x:x['Subject'].nunique()).reset_index(name='# Genres')

director_genre.sort_values(by='# Genres',ascending=False,inplace=True)

top_director_genre=director_genre[:10]

fig = px.scatter(top_director_genre, x='Director', y='# Genres',

                size='# Genres',color='Director',

                 hover_data=['Director'],

                 title='Top Directors of the 20th century who have worked in multiple Genres')

fig.show()

dir_movie=movie[movie['Director'].isin(top_director_genre.Director.tolist())]

director_genre=dir_movie.groupby(['Director','Subject']).apply(lambda x:x['Title'].count()).reset_index(name='# Genres wise Movies')

plt.figure(figsize=(8,10))

pivot_director=pd.pivot(director_genre,index='Director',columns='Subject',values='# Genres wise Movies')

ax=sns.heatmap(pivot_director,annot=True,fmt='g',cmap='autumn_r')

plt.xlabel('Genres',size=20)

plt.ylabel('Directors',size=25)

plt.title('Genre wise filmography of the most verstaile Directors',size=20)
act_movies=movie.groupby('Actor').apply(lambda x:x['Title'].count()).reset_index(name='Acted in Movies')

actr_movies=movie.groupby('Actress').apply(lambda x:x['Title'].count()).reset_index(name='Acted in Movies')

actr_movies.columns=['Actor','Acted in Movies']

acted_movies=pd.concat([act_movies,actr_movies])

acted_movies=acted_movies[acted_movies['Acted in Movies']>1]

dir_movies=movie.groupby('Director').apply(lambda x:x['Title'].count()).reset_index(name='Directed Movies')

act_dir_movies=pd.merge(acted_movies,dir_movies,how='inner',left_on='Actor',right_on='Director')

act_dir_movies=act_dir_movies[act_dir_movies['Directed Movies']>1]

fig = px.scatter(act_dir_movies, x='Acted in Movies', y='Directed Movies',color='Acted in Movies',size='Acted in Movies',

                 hover_data=['Actor'],

                 title='Artists who have acted and directed more than 1 movies')

fig.show()
award_movies=movie[movie['Awards']=='Yes']

no_award_movies=movie[movie['Awards']!='Yes']

award_filter=movie[movie['Awards'].isnull()==False]

award_filter=award_filter[award_filter['Length'].isnull()==False]

award_filter=award_filter[award_filter['Year'].isnull()==False]

award_time_yr=award_filter.groupby(['Year','Awards']).apply(lambda x:np.average(x['Length'])).reset_index(name='Movie Time')

fig = px.line(award_time_yr, x="Year", y="Movie Time", color='Awards',title='Movie Time impacting Awards over the Years')

fig.show()
award_filter=movie[movie['Awards'].isnull()==False]

award_filter=award_filter[award_filter['Popularity'].isnull()==False]

award_filter=award_filter[award_filter['Year'].isnull()==False]

award_time_yr=award_filter.groupby(['Year','Awards']).apply(lambda x:np.average(x['Popularity'])).reset_index(name='Popularity')

fig = px.scatter(award_time_yr, x="Year", y="Popularity", color='Awards',size='Popularity',title='Popularity affecting the Awards over the years')

fig.show()
actor_award=award_movies.groupby('Actor').apply(lambda x:x['Awards'].count()).reset_index(name='# Awards')

actress_award=award_movies.groupby('Actress').apply(lambda x:x['Awards'].count()).reset_index(name='# Awards')

director_award=award_movies.groupby('Director').apply(lambda x:x['Awards'].count()).reset_index(name='# Awards')

actor_award.sort_values(by="# Awards",ascending=False,inplace=True)

actress_award.sort_values(by="# Awards",ascending=False,inplace=True)

director_award.sort_values(by="# Awards",ascending=False,inplace=True)

top_actor_award=actor_award[:10]

top_actor_award['Category']='Actor'

top_actress_award=actress_award[0:10]

top_actress_award['Category']='Actress'

top_director_award=director_award[0:10]

top_director_award['Category']='Director'

top_actor_award.columns=['Artist',"# Awards",'Category']

top_actress_award.columns=['Artist',"# Awards",'Category']

top_director_award.columns=['Artist',"# Awards",'Category']

top_artist_award=pd.concat([top_actor_award,top_actress_award,top_director_award])

fig = px.bar(top_artist_award, x='Artist', y="# Awards",

                color='Category',

                 hover_data=['Artist'],

                 title='Top Artists of the 20th century who have won most Awards')

fig.show()
genre_award=movie.groupby(['Subject','Awards']).apply(lambda x:x['Title'].count()).reset_index(name='Counts')

piv_genre_count=pd.pivot(genre_award,index='Subject',columns='Awards',values='Counts')

plt.figure(figsize=(4,8))

sns.heatmap(piv_genre_count,annot=True,fmt='g',cmap='inferno_r')

plt.xlabel('Won Awards?',size=15)

plt.ylabel('Genres',size=15)

plt.title('Summary of Awards won by the Genres',size=20)