import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv("../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")

genres = dataset['Genres'].str.get_dummies(',')

data = pd.concat([dataset,genres],axis=1,sort=False)

data.drop(['Unnamed: 0'],axis=1,inplace=True)

data.fillna(np.nan,inplace=True)
data.head()
data.columns
#Missing Values

total = data.isnull().sum().sort_values(ascending=False)

percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)

missing = pd.concat([total,percent],axis=1,keys=['Total','Percent'])[:8]

fig = px.bar(missing,x=missing.index,y='Total',title='Missing Values',hover_data=['Percent'],

             labels={'index':'Column'})

fig.update_traces(marker_color= '#faa476')

fig.show()
#Platforms

netflix = len(data[data['Netflix']==1])

hulu = len(data[data['Hulu']==1])

prime = len(data[data['Prime Video']==1])

disney = len(data[data['Disney+']==1])

Platform=['Netflix','Hulu','Prime Video','Disney+']

Count = [netflix,hulu,prime,disney]



fig = px.pie(names=Platform,values=Count,title='Movie Count Of Different Platforms',

            color_discrete_sequence=px.colors.sequential.Sunsetdark)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
netflix_movies = data.loc[data['Netflix'] == 1].drop(['Hulu', 'Prime Video', 'Disney+', 'Type'],axis=1)

hulu_movies = data.loc[data['Hulu'] == 1].drop(['Netflix', 'Prime Video', 'Disney+', 'Type'],axis=1)

prime_video_movies = data.loc[data['Prime Video'] == 1].drop(['Netflix','Hulu', 'Disney+', 'Type'],axis=1)

disney_movies = data.loc[data['Disney+'] == 1].drop(['Netflix','Hulu', 'Prime Video', 'Type'],axis=1)
#Platforms with IMDB 8+ Movies

count_imdb = [len(netflix_movies[netflix_movies['IMDb']>8]),len(hulu_movies[hulu_movies['IMDb']>8]),

              len(prime_video_movies[prime_video_movies['IMDb']>8]),len(disney_movies[disney_movies['IMDb']>8])]

platform = ['Netflix','Hulu','Prime Video','Disney+']



top_rated = pd.DataFrame({'Platforms':platform,'Count':count_imdb})

fig = px.bar(top_rated,x='Platforms',y='Count',color='Count',color_continuous_scale='Sunsetdark',title='IMDB 8+ Movies on different Platforms')

fig.show()
#Top Movies on Different Platforms

n = netflix_movies.sort_values('IMDb',ascending=False).head(10)

h = hulu_movies.sort_values('IMDb',ascending=False).head(10)

p = prime_video_movies.sort_values('IMDb',ascending=False).head(10)

d = disney_movies.sort_values('IMDb',ascending=False).head(10)



fig = make_subplots(rows=4, cols=1,subplot_titles=("Top 10 Movies on Netflix","Top 10 Movies on Hulu",

                                                   "Top 10 Movies on Prime Video","Top 10 Movies on Disney"))



fig.add_trace(go.Bar(y=n['Title'],x=n['IMDb'],orientation='h',marker=dict(color=n['IMDb'],coloraxis="coloraxis"))

             ,row=1,col=1)

fig.add_trace(go.Bar(y=h['Title'],x=h['IMDb'],orientation='h',marker=dict(color=h['IMDb'], coloraxis="coloraxis")),row=2,col=1)

fig.add_trace(go.Bar(y=p['Title'],x=p['IMDb'],orientation='h',marker=dict(color=p['IMDb'], coloraxis="coloraxis")),row=3,col=1)

fig.add_trace(go.Bar(y=d['Title'],x=d['IMDb'],orientation='h',marker=dict(color=d['IMDb'], coloraxis="coloraxis")),row=4,col=1)



fig.update_layout(height=1300, width=1000, title_text="Top Movies on Different Platforms based on IMDB Rating",

                  coloraxis=dict(colorscale='Sunsetdark'),showlegend=False)

fig.show()
#Avg Runtime om Different Platforms

avg_runtime = [netflix_movies['Runtime'].mean(),hulu_movies['Runtime'].mean(),prime_video_movies['Runtime'].mean(),

               disney_movies['Runtime'].mean()]

platform = ['Netflix','Hulu','Prime Video','Disney+']



runtime_ott = pd.DataFrame({'Platforms':platform,'Avg Runtime': avg_runtime})

fig = px.bar(runtime_ott,x='Platforms',y='Avg Runtime',color='Avg Runtime',color_continuous_scale='Sunsetdark',title='Avg Runtime on different Platforms')

fig.show()
#Year

year_count = data.groupby('Year')['Title'].count()

year_movie = data.groupby('Year')[['Netflix','Hulu','Prime Video','Disney+']].sum()

year_data = pd.concat([year_count,year_movie],axis=1).reset_index().rename(columns={'Title':'Movie Count'})



fig = px.bar(year_data,x='Year',y='Movie Count',hover_data=['Netflix','Hulu','Prime Video','Disney+'],

             color='Movie Count',color_continuous_scale='Sunsetdark',title='Movie Count By Year')

fig.show()
#Best movie every year

best_movie_year = data.sort_values('IMDb',ascending=False).groupby('Year').first().reset_index()

fig = px.scatter(best_movie_year,x='Year',y='IMDb',hover_data=['Title','Runtime','Genres','Language'],

                 color_continuous_scale='Sunsetdark',color='IMDb',size='IMDb',

                 title='Best Movie Each Year According to IMDB Rating')

fig.show()
#Movie Count by Language

lan_count = data.groupby('Language')['Title'].count()

lan_movie = data.groupby('Language')[['Netflix','Hulu','Prime Video','Disney+']].sum()

language_data = pd.concat([lan_count,lan_movie],axis=1).reset_index().rename(columns={'Title':'Movie Count'})

language_data = language_data.sort_values('Movie Count',ascending=False)[:10]



fig = px.bar(language_data,x='Language',y='Movie Count',hover_data=['Netflix','Hulu','Prime Video','Disney+'],color='Movie Count',color_continuous_scale='Sunsetdark',title='Movie Count by Language')

fig.show()
#best movie in top 10 languages

best_movie_lang = data.sort_values('IMDb',ascending=False).groupby('Language')[['Title','IMDb','Runtime','Genres','Directors']].first().reset_index()

best_movie_lang['Count'] = lan_count.reset_index()['Title']



best_movie_lang = best_movie_lang.sort_values('Count',ascending=False).head(10)



fig = px.scatter(best_movie_lang,x='Language',y='IMDb',hover_data=['Title','Runtime','Genres','Directors'],

                 color_continuous_scale='Sunsetdark',color='IMDb',size='IMDb',

                 title='Best Movie In Top 10 Languages According to IMDB Rating')

fig.show()
#Country

country_count = data.groupby('Country')['Title'].count()

country_movie = data.groupby('Country')[['Netflix','Hulu','Prime Video','Disney+']].sum()

country_data = pd.concat([country_count,country_movie],axis=1).reset_index().rename(columns={'Title':'Movie Count'})

country_data = country_data.sort_values('Movie Count',ascending=False)[:10]



fig = px.bar(country_data,x='Country',y='Movie Count',hover_data=['Netflix','Hulu','Prime Video','Disney+'],

             color='Movie Count',color_continuous_scale='Sunsetdark',title='Movie Count by Country')

fig.show()
#best movie in top 10 Countries

best_movie_con = data.sort_values('IMDb',ascending=False).groupby('Country')[['Title','IMDb','Runtime','Genres','Year']].first().reset_index()

best_movie_con['Count'] = country_count.reset_index()['Title']



best_movie_con = best_movie_con.sort_values('Count',ascending=False).head(10)



fig = px.scatter(best_movie_con,x='Country',y='IMDb',hover_data=['Title','Runtime','Genres','Year'],

                 color_continuous_scale='Sunsetdark',color='IMDb',size='IMDb',

                 title='Best Movie In Top 10 Countries According to IMDB Rating')

fig.show()
#Runtime

fig = px.histogram(data,x='Runtime',opacity=0.8)

fig.update_traces(marker_color='#faa476')

fig.show()
# Top Movie with longest 

long_movie = data.sort_values('Runtime',ascending=False).reset_index(drop=True).head(10)

long_movie.fillna("NA",inplace=True)

for x in ['Netflix','Hulu','Prime Video','Disney+']:

    long_movie[x].replace(1,x,inplace=True)

    long_movie[x].replace(0,"",inplace=True)



long_movie['Platform']=long_movie[['Netflix','Hulu','Prime Video','Disney+']].agg("  ".join,axis=1)

  

fig = px.bar(long_movie,x='Title',y='Runtime',hover_data=['Directors','IMDb','Rotten Tomatoes','Platform'],title='Top 10 Longest Movies',color='Runtime', color_continuous_scale='Sunsetdark')

fig.show()
#Highest Rating

imdb_rating = data.sort_values('IMDb',ascending=False).reset_index(drop=True).head(10)

imdb_rating.fillna("NA",inplace=True)

for x in ['Netflix','Hulu','Prime Video','Disney+']:

    imdb_rating[x].replace(1,x,inplace=True)

    imdb_rating[x].replace(0,"",inplace=True)



imdb_rating['Platform']=imdb_rating[['Netflix','Hulu','Prime Video','Disney+']].agg("  ".join,axis=1)



fig = px.bar(imdb_rating,x='Title',y='Runtime',hover_data=['Directors','IMDb','Platform'],title='Top 10 Highest IMDB Rated Movies',

             color='Runtime', color_continuous_scale='Sunsetdark')

fig.show()
dir_count = data.groupby('Directors')['Title'].count()

dir_movie = data.groupby('Directors')[['Netflix','Hulu','Prime Video','Disney+']].sum()

dir_rating = data.groupby('Directors')['IMDb'].mean()

dir_data = pd.concat([dir_count,dir_movie,dir_rating],axis=1).reset_index().rename(columns={'Title':'Movie Count',

                                                                                            'IMDb':'Avg Rating'})

dir_count_data = dir_data.sort_values('Movie Count',ascending=False).head(10)



fig = px.bar(dir_count_data,x='Directors',y='Movie Count',hover_data=['Netflix','Hulu','Prime Video','Disney+'],color='Movie Count',color_continuous_scale='Sunsetdark',title='Top 10 Directors Movie Count')

fig.show()
jc = data[data['Directors']=='Jay Chapman'].sort_values('IMDb',ascending=False).head(5)

jk = data[data['Directors']=='Joseph Kane'].sort_values('IMDb',ascending=False).head(5)

cc = data[data['Directors']=='Cheh Chang'].sort_values('IMDb',ascending=False).head(5)

jw = data[data['Directors']=='Jim Wynorski'].sort_values('IMDb',ascending=False).head(5)

sn = data[data['Directors']=='Sam Newfield'].sort_values('IMDb',ascending=False).head(5)



fig = make_subplots(rows=5,cols=1,subplot_titles=("Top 5 Movies of Jay Chapman",

                                                  "Top 5 Movies of Joseph Kane",

                                                  "Top 5 Movies of Cheh Chang",

                                                  "Top 5 Movies of Jim Wynorski",

                                                  "Top 5 Movies of Sam Newfield"))



fig.add_trace(go.Bar(x=jc['IMDb'],y=jc['Title'],orientation='h',marker=dict(color=jc['IMDb'],coloraxis="coloraxis")),row=1,col=1)

fig.add_trace(go.Bar(x=jk['IMDb'],y=jk['Title'],orientation='h',marker=dict(color=jk['IMDb'],coloraxis="coloraxis")),row=2,col=1)

fig.add_trace(go.Bar(x=cc['IMDb'],y=cc['Title'],orientation='h',marker=dict(color=cc['IMDb'],coloraxis="coloraxis")),row=3,col=1)

fig.add_trace(go.Bar(x=jw['IMDb'],y=jw['Title'],orientation='h',marker=dict(color=jw['IMDb'],coloraxis="coloraxis")),row=4,col=1)

fig.add_trace(go.Bar(x=sn['IMDb'],y=sn['Title'],orientation='h',marker=dict(color=sn['IMDb'],coloraxis="coloraxis")),row=5,col=1)



fig.update_layout(height=1500, width=1000, title_text="Top 5 Directors With Most Movies",

                  coloraxis=dict(colorscale='Sunsetdark'),showlegend=False)

fig.show()
dir_rating_data = dir_data.sort_values('Avg Rating',ascending=False).head(10)

fig = px.bar(dir_rating_data,x='Directors',y='Avg Rating',hover_data=['Movie Count','Netflix','Hulu','Prime Video','Disney+'],color='Avg Rating',color_continuous_scale='Sunsetdark',title='Top 10 Directors with Movies Having Highest IMDB Rating')

fig.show()
gen_count = data.groupby('Genres')['Title'].count()

gen_movie = data.groupby('Genres')[['Netflix','Hulu','Prime Video','Disney+']].sum()

gen_data = pd.concat([gen_count,gen_movie],axis=1).reset_index().rename(columns={'Title':'Movie Count'})

gen_data = gen_data.sort_values('Movie Count',ascending=False)[:10]



fig = px.bar(gen_data,x='Genres',y='Movie Count',hover_data=['Netflix','Hulu','Prime Video','Disney+'],color='Movie Count',color_continuous_scale='Sunsetdark',title='Top 10 Genres Movie Count')

fig.show()
drama = data[data['Drama']==1].sort_values('IMDb',ascending=False).head(10)

documentary = data[data['Documentary']==1].sort_values('IMDb',ascending=False).head(10)

comedy = data[data['Comedy']==1].sort_values('IMDb',ascending=False).head(10)



fig = make_subplots(rows=3,cols=1,subplot_titles=("Top 10 Movies of Drama Genre","Top 10 Movies of Documentary Genre",

                                                  "Top 10 Movies of Comedy Genre"))

fig.add_trace(go.Bar(x=drama['IMDb'],y=drama['Title'],orientation='h',marker=dict(color=drama['IMDb'],coloraxis="coloraxis")),row=1,col=1)

fig.add_trace(go.Bar(x=documentary['IMDb'],y=documentary['Title'],orientation='h',marker=dict(color=documentary['IMDb'],coloraxis="coloraxis")),row=2,col=1)

fig.add_trace(go.Bar(x=comedy['IMDb'],y=comedy['Title'],orientation='h',marker=dict(color=comedy['IMDb'],coloraxis="coloraxis")),row=3,col=1)



fig.update_layout(height=1000, width=1000, title_text="Top Movies of Different Genres based on IMDB Rating",

                  coloraxis=dict(colorscale='Sunsetdark'),showlegend=False)

fig.show()
#Select the features on the basis of ehich you want to cluster

features = data[['Action', 'Adventure', 'Animation',

                 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',

                 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music',

                 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi',

                 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']].astype(int)



#Scaling the data

scaler = StandardScaler()

scaled_data = scaler.fit_transform(features)



#Using TSNE

tsne = TSNE(n_components=2)

transformed_genre = tsne.fit_transform(scaled_data)
#KMeans - Elbow Method

distortions = []

K = range(1,100)

for k in K:

    kmean = KMeans(n_clusters=k)

    kmean.fit(scaled_data)

    distortions.append(kmean.inertia_)

fig = px.line(x=K,y=distortions,title='The Elbow Method Showing The Optimal K',

              labels={'x':'No of Clusters','y':'Distortions'})

fig.show()
#Kmeans

cluster = KMeans(n_clusters=27)

group_pred = cluster.fit_predict(scaled_data)



tsne_df = pd.DataFrame(np.column_stack((transformed_genre,group_pred,data['Title'],data['Genres'])),columns=['X','Y','Group','Title','Genres'])



fig = px.scatter(tsne_df,x='X',y='Y',hover_data=['Title','Genres'],color='Group',

                 color_discrete_sequence=px.colors.cyclical.IceFire)

fig.show()