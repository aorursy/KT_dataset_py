import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import json

import seaborn as sns

%matplotlib inline

import re

from wordcloud import WordCloud
import plotly

plotly.tools.set_credentials_file(username='SakshamVikram', api_key='QoqXbiYSFGnkC40ovvMW')
import plotly.plotly as py

import plotly.graph_objs as go

movies=pd.read_csv("../input/tmdb_5000_movies.csv")

credits=pd.read_csv("../input/tmdb_5000_credits.csv")

movie_json=['genres','keywords','production_countries','production_companies','spoken_languages',]

for col in movie_json:

    movies[col]=movies[col].apply(json.loads)

movies['release_date']=pd.to_datetime(movies['release_date'])



credits_json=['cast','crew']

for col in credits_json:

    credits[col]=credits[col].apply(json.loads)





    
l=0

for i in range(len(movies.title)):

 

    if not movies.title[i]==movies.original_title[i]:

                   print([movies.title.iloc[i],movies.original_title.iloc[i]])

                   l+=1

    if l>10:

        break

                    

                   

                    

               

               
TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {

    'budget': 'budget',

    'genres': 'genres',

    'revenue': 'gross',

    'title': 'movie_title',

    'runtime': 'duration',

    'original_language': 'language',  # it's possible that spoken_languages would be a better match

    'keywords': 'plot_keywords',

    'vote_count': 'num_voted_users',

                                         }



IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}





def safe_access(container, index_values):

    # return a missing value rather than an error upon indexing/key failure

    result = container

    try:

        for idx in index_values:

            result = result[idx]

        return result

    except IndexError or KeyError:

        return pd.np.nan





def get_director(crew_data):

    directors = [x['name'] for x in crew_data if x['job'] == 'Director']

    return safe_access(directors, [0])





def pipe_flatten_names(keywords):

    return '|'.join([x['name'] for x in keywords])





def convert_format(movies, credits):

    # Converts TMDb data to make it as compatible as possible with kernels built on the original version of the data.

    tmdb_movies = movies.copy()

    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)

    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)

    # I'm assuming that the first production country is equivalent, but have not been able to validate this

    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['director_name'] = credits['crew'].apply(get_director)

    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))

    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))

    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)

    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)

    tmdb_movies['actor_1_gender']=credits['cast'].apply(lambda x:safe_access(x,[0,'gender']))

    tmdb_movies['actor_2_gender']=credits['cast'].apply(lambda x:safe_access(x,[1,'gender']))

    tmdb_movies['actor_3_gender']=credits['cast'].apply(lambda x:safe_access(x,[2,'gender']))

    

    

    return tmdb_movies
tmdb_movies=convert_format(movies,credits)

cleaned=tmdb_movies.drop(['homepage','tagline'],axis=1).dropna(axis=0).reset_index()
tmdb_movies.head(5)
numerical_col=['budget','popularity','gross','num_voted_users','duration']

s=cleaned.groupby('title_year').count()

plt.figure(figsize=(25,12))

sns.pointplot(x=np.array(s.index),y=s['budget'])

sns.regplot(x=np.array(s.index),y=s['budget'],scatter=False,lowess=True)

plt.xticks(x=s.index,rotation='vertical',size='x-large')

plt.yticks(size='x-large')



plt.xlabel("years",size=30,weight='heavy',color='red')

plt.ylabel('No. of Movies',weight='heavy',color='red',size=30)

plt.title('No. of Movies Released By year',size=50)



plt.show()

from sklearn import preprocessing

wm=lambda x:np.average(x,weights=cleaned.loc[x.index,'num_voted_users'])

data=cleaned.groupby('title_year').aggregate({'vote_average':wm,'num_voted_users':lambda x:np.sum(x)})

scaler=preprocessing.MinMaxScaler((0,10))##Scaling to bring both the columns on same sacle

scaled=pd.DataFrame(scaler.fit_transform(data),columns=data.columns)

plt.figure(figsize=(25,12))

plt.plot(np.array(data.index),scaled['vote_average'],'r')

plt.plot(np.array(data.index),scaled['num_voted_users'],'b')

plt.xticks(x=data.index,rotation='vertical',size='x-large')

plt.yticks(size='x-large')



plt.xlabel("years",size=30,weight='heavy',color='red')



plt.legend(loc='upper left',fontsize=20)





plt.show()
cleaned.columns
##This is the popularity of directors By The Votes

great_directors=cleaned.groupby('director_name').agg({'gross':np.average,'vote_average':np.average})

g_gross=great_directors.sort_values('gross',ascending=False).reset_index().iloc[:10,:]

g_vote=great_directors.sort_values('vote_average',ascending=False).reset_index().iloc[:10,:]



fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(25,12))

sns.barplot(y='director_name',x='vote_average',data=g_vote,ax=ax[0])

sns.barplot(y='director_name',x='gross',data=g_gross,ax=ax[1])

ax[0].set_yticklabels(labels=g_vote['director_name'],size='x-large')

ax[0].set_ylabel('Directors',size=20)

ax[0].set_xlabel('Vote_Average',size=20)

ax[0].set_title('Critically Aclaimed Directors',fontsize=30)

ax[1].set_yticklabels(labels=g_gross['director_name'],size='x-large')

ax[1].set_ylabel('Directors',size=20)

ax[1].set_xlabel('Gross',size=20)

ax[1].set_title('BlockBuster Promising Directors',fontsize=30)



plt.show()
new_data=cleaned[['budget','popularity','gross','num_voted_users','duration','vote_average']]

col=['budget', 'popularity', 'gross', 'voter_count', 'duration',

       'vote_average']

corr=new_data.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr,annot=True,square=True,linewidths=.15)

plt.xticks(label=col,size='x-large',rotation='vertical')

plt.yticks(y=col,size='x-large',rotation='horizontal')

plt.show()
df=cleaned[(cleaned['actor_1_gender']==1)&(cleaned['actor_2_gender']==2)&(cleaned['actor_3_gender']==2)][['actor_1_name','gross','popularity']].reset_index()

df1=cleaned[(cleaned['actor_1_gender']==1)&(cleaned['actor_2_gender']==1)&(cleaned['actor_3_gender']==2)][['actor_2_name','gross','popularity']].reset_index()

df2=cleaned[(cleaned['actor_1_gender']==1)&(cleaned['actor_2_gender']==1)&(cleaned['actor_3_gender']==1)][['actor_3_name','gross','popularity']].reset_index()
df.columns=['index','Actress','gross','popularity']

df1.columns=['index','Actress','gross','popularity']

df2.columns=['index','Actress','gross','popularity']

s=pd.concat([df,df1,df2],axis=0)

s.drop('index',inplace=True,axis=1)

s=s[s['gross']!=0]

pro=s.groupby('Actress').agg({'gross':np.average}).reset_index()

grosser=pro.sort_values('gross',ascending=False).reset_index().drop('index',axis=1).iloc[:10,:]

pro=s.groupby('Actress').agg({'popularity':np.average}).reset_index()

popular=pro.sort_values('popularity',ascending=False).reset_index().drop('index',axis=1).iloc[:10,:]



fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(25,12))

sns.barplot(y='Actress',x='popularity',data=popular,ax=ax[0])

sns.barplot(y='Actress',x='gross',data=grosser,ax=ax[1])

ax[0].set_yticklabels(labels=popular['Actress'],size='x-large')

ax[0].set_ylabel('Actress',size=20)

ax[0].set_xlabel('Popularity',size=20)

ax[0].set_title('Most Popular Female Protagonists',fontsize=30)

ax[1].set_yticklabels(labels=grosser['Actress'],size='x-large')

ax[1].set_ylabel('Actress',size=20)

ax[1].set_xlabel('Gross',size=20)

ax[1].set_title('BlockBuster Promising Actresses',fontsize=30)



plt.show()
temp=""

for i in cleaned.genres:

    temp=temp+'|'+i

    

genres_list=list(set(temp.split('|')))

del genres_list[0]##Data Cleaning.

genres_list[:10]

dict_genres=dict()

for i in genres_list:

    dict_genres[i]=[]



for i in range(len(cleaned.genres)):

    diff=cleaned.genres[i].split("|")

    for j in diff:

        if re.search(r'[a-zA-Z]+',j):

            dict_genres[j].append(cleaned.popularity[i])

 

genres=dict_genres.keys()

favourite_mean=[]

for i in genres:

    favourite_mean.append((np.array(dict_genres[i]).mean(),np.array(dict_genres[i]).std()/np.sqrt(len(dict_genres[i])),i))

top_10=sorted(favourite_mean,key=lambda x:x[0],reverse=True)

means,std,genres=zip(*top_10)



plt.figure(figsize=(25,10))

x = np.arange(1,21,1)

y = means

labels = list(genres)

plt.errorbar(x, y,std,linestyle='None', marker='^')



plt.xticks(x, labels,rotation='vertical',size='x-large')

plt.yticks(size='x-large')

plt.show()
list(genres)[:10]
labels=list(genres)[:10]

values=means[:10]

fig1, ax1 = plt.subplots()

ax1.pie(values, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
dict_freq={}



for i in genres:

    dict_freq[i]=len(dict_genres[i])

wordcloud = WordCloud()

wordcloud.generate_from_frequencies(frequencies=dict_freq)

plt.figure(figsize=(12,25))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()