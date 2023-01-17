import pandas as pd # package for high-performance, easy-to-use data 

#structures and data analysis

import numpy as np # fundamental package for scientific computing with Python

import matplotlib

import matplotlib.pyplot as plt # for plotting

import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.offline as offline

offline.init_notebook_mode()

from pylab import rcParams







# import cufflinks and offline mode

import cufflinks as cf

cf.go_offline()



# from sklearn import preprocessing

# # Supress unnecessary warnings so that presentation looks clean

import warnings

warnings.filterwarnings("ignore")



from IPython.display import Image
ott=pd.read_csv('../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
ott.head()
ott.columns
ott.shape
ott['Country'].value_counts()
ott.dtypes
plt.figure(figsize=(12,8))

corr = ott.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(10, 8))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap='magma', annot=True, fmt=".2f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
def missing_percentage(data):

    

    """

    A function for returning missing ratios.

    """

    

    total = ott.isnull().sum().sort_values(

        ascending=False)[ott.isnull().sum().sort_values(ascending=False) != 0]

    percent = (ott.isnull().sum().sort_values(ascending=False) / len(ott) *

               100)[(data.isnull().sum().sort_values(ascending=False) / len(ott) *

                     100) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing = missing_percentage(ott)



fig, ax = plt.subplots(figsize=(20, 5))

sns.barplot(x=missing.index, y='Percent', data=missing, palette='Reds_r')

plt.xticks(rotation=90)



display(missing.T.style.background_gradient(cmap='Reds', axis=1))
ott.columns
ott.drop(['Rotten Tomatoes','Unnamed: 0','Type','ID','Age'],axis=1, inplace=True)

ott.head()
ott['Language'].value_counts()
ott.groupby('Genres').IMDb.agg(['count','max','min','mean'])
# rating distibution 

rcParams['figure.figsize'] = 10,8

g = sns.kdeplot(ott.IMDb, color="Red", shade = True)

g.set_xlabel("Rating")

g.set_ylabel("Frequency")

plt.title('Ratings Range',size = 15)
# rating distibution 

rcParams['figure.figsize'] = 10,8

g = sns.kdeplot(ott.Runtime, color="Red", shade = True)

g.set_xlabel("Runtime Stretch")

g.set_ylabel("Frequency")

plt.title('Ratings Range',size = 15)
total_movies_Netflix = len(ott[ott['Netflix'] == 1].index)

total_movies_Hulu = len(ott[ott['Hulu'] == 1].index)

total_movies_Prime =len(ott[ott['Prime Video'] == 1].index)

total_movies_Disney = len(ott[ott['Disney+'] == 1].index)





print(total_movies_Netflix)

print(total_movies_Hulu)

print(total_movies_Prime)

print(total_movies_Disney)
tags=['Netflix','Hulu', 'Prime Video','Disney+']

counts=[total_movies_Netflix,total_movies_Hulu,total_movies_Prime,total_movies_Disney]

ott_platform = pd.DataFrame(

    {'Platform': tags,

     'MovieCount': counts,

    })

ott_platform
fig = px.pie(ott_platform,names='Platform', values='MovieCount')

fig.update_traces(rotation=45,pull=[0.1,0.03,0.03,0.03,0.03],title='MOVIE DISTRIBUTION ACROSS PLATFORMS')

fig.show()
movies_with_longer_runtime = ott.sort_values('Runtime',ascending = False).head(15)

fig = px.bar(movies_with_longer_runtime, x='Title', y='Runtime', color='Runtime', height=700)

fig.show()
Image("../input/colorado-th/colorado.JPG")
year_wise_movie_release= ott.groupby('Year')['Title'].count().reset_index().rename(columns = {'Title':'Number_of_Movies'})

fig = px.bar(year_wise_movie_release, x='Year', y='Number_of_Movies', color='Number_of_Movies', height=500)

fig.show()
Image("../input/raja-harishch/rjha.JPG")
favourite_genres = ott.groupby('Genres')['Title'].count().reset_index().rename(columns = {'Title':'Number_of_Movies'}).sort_values('Number_of_Movies',ascending = False).head(15)

fig = px.bar(favourite_genres, x='Genres', y='Number_of_Movies', color='Number_of_Movies', height=700)

fig.show()
top_15_directors = ott.groupby('Directors')['Title'].count().reset_index().rename(columns = {'Title':'Number_of_Movies'}).sort_values('Number_of_Movies',ascending = False).head(15)

fig = px.bar(top_15_directors, x='Directors', y='Number_of_Movies', color='Number_of_Movies', height=650)

fig.show()
Image("../input/jaychapman/jaychapman.JPG")
top_15_countries = ott.groupby('Country')['Title'].count().reset_index().rename(columns = {'Title':'Number_of_Movies'}).sort_values('Number_of_Movies',ascending = False).head(15)

fig = px.bar(top_15_countries, x='Country', y='Number_of_Movies', color='Number_of_Movies', height=650)

fig.show()
best_15_works = ott.sort_values('IMDb',ascending = False).head(15)

fig = px.bar(best_15_works, x='Title', y='IMDb', color='IMDb', height=600)

fig.show()
Image("../input/srkdavid/srkdavid.JPG")
Image("../input/natsamrat/nstrm.JPG")
genre_runtime = ott.sort_values('Runtime',ascending = False).head(15)

fig = px.bar(genre_runtime, x='Genres', y='Runtime', color='IMDb', height=600)

fig.show()
genre_runtime = ott.sort_values('Runtime',ascending = False).head(15)

fig = px.bar(genre_runtime, x='Language', y='Runtime', color='IMDb', height=600)

fig.show()
Image("../input/thatsallfolks/thatsall.JPG")