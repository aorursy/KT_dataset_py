# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import collections
%matplotlib inline

netflix=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
netflix.head()
netflix["date_added"] = pd.to_datetime(netflix['date_added'])
netflix['year_added'] = netflix['date_added'].dt.year
## Shape of dataset
netflix.shape
## Columns in the dataframe
netflix.columns
## Checking for Null Values
netflix.isnull().sum()
## Check if there are any duplicate Titles
netflix.duplicated().sum()
## Create duplicate dataset

netflix_copy = netflix.copy()
netflix_copy.head()
netflix_copy = netflix_copy.dropna()
netflix_copy.shape
## Derive new columns from date which will provide the day, month and year in which they were added in the service

netflix_copy['date_added'] = pd.to_datetime(netflix['date_added'])
netflix_copy['Day_of_release'] = netflix_copy['date_added'].dt.day
netflix_copy['Month_of_release']= netflix_copy['date_added'].dt.month
netflix_copy['Year_of_release'] = netflix_copy['date_added'].dt.year

netflix_copy['Year_of_release'].astype(int);
netflix_copy['Day_of_release'].astype(int);
col = "type"
group_value = netflix[col].value_counts().reset_index()
group_value = group_value.rename(columns = {col : "count", "index" : col})

## plotting graph

labels = group_value.type
sizes = group_value['count']
explode=(0.1,0)

fig1, ax = plt.subplots()
ax.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%', shadow=True)
ax.axis('equal')
plt.show()
plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(x="rating", data=netflix, palette="Set1", order=netflix['rating'].value_counts().index[0:15])
# Make separate dataframe for Movies and shows

netflix_shows=netflix[netflix['type']=='TV Show']
netflix_movies=netflix[netflix['type']=='Movie']
plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(y="release_year", data=netflix_movies, palette="Set2", order=netflix_movies['release_year'].value_counts().index[0:15])
plt.title('Movies released per year')
plt.xlabel('Number of movies released')
plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(y="release_year", data=netflix_shows, palette="Set2", order=netflix_shows['release_year'].value_counts().index[0:15])
plt.title('TV Shows released per year')
plt.xlabel('Number of TV Shows released')
imdb_ratings=pd.read_csv('/kaggle/input/imdb-extensive-dataset/IMDb ratings.csv',usecols=['weighted_average_vote'])
imdb_titles=pd.read_csv('/kaggle/input/imdb-extensive-dataset/IMDb movies.csv', usecols=['title','year','genre','language'])

ratings = pd.DataFrame({'Title':imdb_titles.title,
                    'Release Year':imdb_titles.year,
                    'Language': imdb_titles.language,
                    'Rating': imdb_ratings.weighted_average_vote,
                    'Genre':imdb_titles.genre})
ratings.drop_duplicates(subset=['Title','Release Year','Rating'], inplace=True)
ratings.rename(columns ={'Rating':'Out_of_10_rating'},inplace=True)
ratings.head()
## Drop NA from ratings

ratings = ratings.dropna()
## Using Inner Join to connect Netflix database with the IMDB Ratings

Netflix_Ratings = ratings.merge(netflix, left_on = 'Title', right_on = 'title', how='inner')
Netflix_Ratings.sort_values(by = 'Out_of_10_rating', ascending = False).head()
# Make separate dataframe for Movies and shows

netflix_shows=Netflix_Ratings[Netflix_Ratings['type']=='TV Show'].sort_values(by = 'Out_of_10_rating', ascending = False)
netflix_movies=Netflix_Ratings[Netflix_Ratings['type']=='Movie'].sort_values(by = 'Out_of_10_rating', ascending = False)
top_10_movies = netflix_movies.sort_values("Out_of_10_rating", ascending = False)
top_10_movies = top_10_movies[ top_10_movies['Release Year'] > 2000]
top_10_movies[['title', "Out_of_10_rating"]][0:10]
top_10_shows = netflix_shows.sort_values("Out_of_10_rating", ascending = False)
top_10_shows = top_10_shows[ top_10_shows['Release Year'] > 2000]
top_10_shows[['title', "Out_of_10_rating"]][0:10]
df = Netflix_Ratings[ (Netflix_Ratings['release_year']>2007) & (Netflix_Ratings['release_year']< 2020) ]

d1 = df[df["type"] == "TV Show"]
d2 = df[df["type"] == "Movie"]

col = "release_year"

vc1 = d1[col].value_counts().reset_index()
vc1 = vc1.rename(columns = {col : "count", "index" : col})
vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
vc1 = vc1.sort_values(col)

vc2 = d2[col].value_counts().reset_index()
vc2 = vc2.rename(columns = {col : "count", "index" : col})
vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))
vc2 = vc2.sort_values(col)

trace1 = go.Scatter(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))
trace2 = go.Scatter(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))
data = [trace1, trace2]
layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()
Netflix_Ratings['listed_in'] = Netflix_Ratings['listed_in'].str.split(',')
Netflix_Ratings['listed_in'].explode().nunique()
Netflix_Ratings['listed_in'].explode().unique()
## Removing white spaces
genres = Netflix_Ratings['listed_in'].explode()
genres = [genre.strip() for genre in genres]
genre_count = collections.Counter(genres)
print(genre_count.most_common(5))
len(set(genres))
genre_df = pd.DataFrame(genre_count.most_common(5), columns = ['Genre','Count'])
genre_df
plt = sns.barplot(x = 'Genre', y = 'Count', data = genre_df)
plt.set_xticklabels(plt.get_xticklabels(), rotation=45, ha='right')
## Checking null values for country
Netflix_Ratings['country'].isna().sum()
## Create new dataframe where there are no null values for country.
country = Netflix_Ratings[Netflix_Ratings['country'].notna()]
country['country'].isna().sum()
country.head(20)
## There are more than 1 country for some rows separated by (,)
## Hence, separating these values

country['country'] =country['country'].str.split(',')

country['country'].explode().unique()
country['country'].explode().value_counts()
## Remove the space

countries = country['country'].explode()
countries = [country.strip() for country in countries]
country_count = collections.Counter(countries)

print(country_count.most_common(5))
# Visualize top countries
top_countries = country_count.most_common(5)
top_countries_df = pd.DataFrame(top_countries, columns=['country','count'])
top_countries_df
plt = sns.barplot(x="country", y="count",palette="Set1", data=top_countries_df)
plt.set_xticklabels(plt.get_xticklabels(), rotation=45, ha='right')
Netflix_Ratings['Month'] = pd.DatetimeIndex( Netflix_Ratings['date_added']).month_name()
Netflix_Ratings.head(30)
Months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plot = sns.countplot(x="Month",order=Months, data=Netflix_Ratings)

plot.set_xticklabels(plot.get_xticklabels(), rotation=40 , ha="right")
India = country.explode("country")
India = India[India['country']=='India']
def content_over_years(country):
    movie_per_year=[]

    tv_shows_per_year=[]
    for i in range(2008,2020):
        h=netflix.loc[(netflix['type']=='Movie') & (netflix.year_added==i) & (netflix.country==str(country))] 
        g=netflix.loc[(netflix['type']=='TV Show') & (netflix.year_added==i) &(netflix.country==str(country))] 
        movie_per_year.append(len(h))
        tv_shows_per_year.append(len(g))



    trace1 = go.Scatter(x=[i for i in range(2008,2020)],y=movie_per_year,mode='lines+markers',name='Movies')

    trace2=go.Scatter(x=[i for i in range(2008,2020)],y=tv_shows_per_year,mode='lines+markers',name='TV Shows')

    data=[trace1,trace2]

    layout = go.Layout(title="Content added over the years in "+str(country), legend=dict(x=0.1, y=1.1, orientation="h"))

    fig = go.Figure(data, layout=layout)

    fig.show()
countries=['India']

for country in countries:
    content_over_years(str(country))
## Top 10 Hindi Movies

India[India['Language']=='Hindi'].sort_values(by=['Out_of_10_rating'], ascending=False).head(10)
top_10_shows_india = India[India['Language']=='Hindi'].sort_values(by=['Out_of_10_rating'], ascending=False)
top_10_shows_india = top_10_shows_india[ top_10_shows_india['Release Year'] >= 2000]
top_10_shows_india[['title', "Out_of_10_rating"]][0:10]
