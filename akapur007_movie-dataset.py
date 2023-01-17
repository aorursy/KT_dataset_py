import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Loading the datasets
credits= pd.read_csv('../input/tmdb_5000_credits.csv')
movies= pd.read_csv('../input/tmdb_5000_movies.csv')
#Looking the datasets
credits.head()
movies.head()
# Getting the info
credits.info()
movies.info()
# As per above - we have Null values at homepage, runtime, tagline
#Top 5 Rated Movies as per vote average where vote count are greter then 1000
movies[movies.vote_count>1000].sort_values(by='vote_average',ascending=False).head(5).title
#Avevage rating as per the years
movies['release_date'] = pd.to_datetime(movies['release_date'])
movies['Year'] = movies['release_date'].apply(lambda time: time.year)
movies['Month'] = movies['release_date'].apply(lambda time: time.month)
movies['Day of Week'] = movies['release_date'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

movies['Day of Week'] = movies['Day of Week'].map(dmap)
movies.head()
#Group by year
groupByYear_VoteAverage = movies.groupby(by='Year',as_index=False).vote_average.mean().sort_values(by="vote_average",ascending=False)
groupByYear_VoteAverage.head()
#Top 30 years in which average ratings are high
plt.figure(figsize=(15,6))
g = sns.barplot(x='Year',y='vote_average',data=groupByYear_VoteAverage.head(30))
for i in g.get_xticklabels():
    i.set_rotation(45)
#Top movies with with highest profits
movies['profits'] = movies['revenue']-movies['budget']
hp_Df= movies.sort_values('profits',ascending=False).head(5)[['original_title','profits','budget','revenue']]
sns.barplot(x='original_title',y='profits',data=hp_Df,hue='revenue')
#No of releases in a year
movies.status.unique()
statusDf = movies['status'].value_counts()
statusDf
#Release group by Year
groupByYear = movies.groupby(by=['Year','status'], as_index='status')['status'].count().unstack()
groupByYear.fillna(0,inplace=True)
groupByYear.tail()
sns.heatmap(groupByYear.sort_values(by='Year',ascending=False),cmap='coolwarm')
sns.clustermap(groupByYear.sort_values(by='Year',ascending=False),cmap='coolwarm')
#No of movies by an actor

import json
castsList = []
def toJson(x):
    l = json.loads(x)
    castsList.extend(l)
credits['cast'].apply(lambda x:toJson(x));
castsDf = pd.DataFrame(castsList)
castsDf['name'].value_counts().head()
castSeries=castsDf['name'].value_counts()
castDfNew=pd.DataFrame(castSeries,).reset_index()
castDfNew.columns = ['name','movies_count']
uniqueCastDf = castsDf[['name','gender']].drop_duplicates()
uniqueCastDf =pd.merge(castDfNew,uniqueCastDf,on='name')
gender ={0:'Unknown',1:'Female',2:'Male'}
uniqueCastDf['gender']=uniqueCastDf['gender'].map(gender);
#Top 5 Male Cast with most movie
uniqueCastDf[uniqueCastDf['gender']=='Male'].head(5)
#Top 5 Female Cast with most movies
uniqueCastDf[uniqueCastDf['gender']=='Female'].head(5)
sns.countplot(x='gender',data=uniqueCastDf)
uniqueCastDf.head(5)
plt.figure(figsize=(16,7))
t =sns.barplot(x='name',y='movies_count',hue='gender',data=uniqueCastDf.head(50))
for i in t.get_xticklabels():
    i.set_rotation(90)
sns.set()
sns.distplot(uniqueCastDf['movies_count'].head(20))
#Creating a new column genre_list which 
movies[movies['genres'].isnull()] # we all genres
def genresToList(x):
    json=pd.read_json(x)
    if(json.empty):
        return []
    else:
        return json['name'].tolist()
        
        
movies['genres_list']=movies['genres'].apply(lambda x:genresToList(x))
movies.head()
genres = []
def addToList(x):
    genres.extend(x)
    
movies['genres_list'].map(addToList);
genresDf = pd.DataFrame(genres,columns=['Genre'])
genresDf['Genre'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,6))
sns.countplot(x='Genre',data=genresDf,)
#Movies with highest budget
topTenMoviesNyBudget = movies.sort_values(by='budget',ascending=False).head(10)
sns.distplot(topTenMoviesNyBudget['budget'],hist_kws=dict(edgecolor="k", linewidth=1))
sns.jointplot(x='popularity',y='budget',data=topTenMoviesNyBudget)
#Convert Release to day month year
movies['release_date'] = pd.to_datetime(movies['release_date'])

movies.head()

movies.head()
movies.head()
byMonth = movies.groupby('Month').count()
byMonth.head()
byMonth['release_date'].plot()

sns.lmplot(x='Month',y='release_date',data=byMonth.reset_index())
rel = movies.groupby(by=['Day of Week','Year']).count()['release_date'].unstack()
rel.head()
rel.fillna(0,inplace=True)
rel.head()
plt.figure(figsize=(20,10))
sns.heatmap(rel)
sns.clustermap(rel,cmap='viridis')

sns.set_style()
sns.distplot(movies.budget,norm_hist=True)
