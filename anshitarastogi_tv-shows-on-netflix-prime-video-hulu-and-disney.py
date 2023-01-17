# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
data.head()
data.drop('Unnamed: 0',axis=1,inplace=True)
data.head()
data.isnull().sum()
data.drop(['Rotten Tomatoes','Age'],axis=1,inplace=True)
data.head()
data.info()
## Movies on each platform ###

platforms=['Netflix','Hulu','Prime Video','Disney+']
movies_netflix=data['Netflix'].sum()
movies_hulu=data['Hulu'].sum()
movies_prime=data['Prime Video'].sum()
movies_disney=data['Disney+'].sum()
platform_count=[movies_netflix,movies_hulu,movies_prime,movies_disney]
print(pd.DataFrame({'Movie_Platform':platforms,'Count':platform_count}))
from plotly.offline import iplot
plot_platform=[go.Pie(
        labels=platforms,
        values=platform_count
)]

layout=go.Layout(title='Plot of Movies on diff Platform')

figure=go.Figure(data=plot_platform)
iplot(figure)
movies_count_by_years=data.groupby('Year')['Title'].count().reset_index().rename(columns={'Title':'Movie Count'})
figure=px.bar(movies_count_by_years,x='Year',y='Movie Count',color='Movie Count',height=600)
figure.show()
#movies_count_by_years
movies_count_by_language=data.groupby('Language')['Title'].count().reset_index().sort_values('Title',ascending=False).head(10).rename(columns={'Title':'Movie Count'})
figure=px.bar(movies_count_by_language,x='Language',y='Movie Count',color='Movie Count',height=600)

figure.show()
movies_count_by_runtime=data.groupby('Runtime')['Title'].count().reset_index().sort_values('Title',ascending=False).rename(columns={'Title':'Movie Count'})
figure=px.bar(movies_count_by_runtime,x='Runtime',y='Movie Count',color='Movie Count',height=400)
figure.show()
#movies_count_by_runtime
lengthiest_movie=data.sort_values(by='Runtime',ascending=False).head(10)
fig=px.bar(lengthiest_movie,x='Title',y='Runtime',height=600,color='Title')
fig.update_layout(title='Lengthiest Movie')
fig.show()
fig=px.bar(lengthiest_movie,x='Title',y='IMDb',height=600,color='Title',range_y=[0,10])
fig.update_layout(title='IMDb Ratings Lengthiest Movie')
fig.show()
movies_count_by_country=data.groupby('Country')['Title'].count().reset_index().sort_values('Title',ascending=False).head(10).rename(columns={'Title':'Count'})
fig=px.pie(movies_count_by_country,names='Country',values='Count')
fig.update_layout(title='Movies Count based on Countries')
fig.show()
movies_count_by_IMDB=data.groupby('IMDb')['Title'].count().reset_index().sort_values('IMDb',ascending=False).head(10).rename(columns={'Title':'Count'})
fig=px.pie(movies_count_by_IMDB,names='IMDb',values='Count')
fig.update_traces(textinfo='percent+label',title='Movies Count based on IMDb Ratings')
fig.show()
data['Genres'].value_counts().head()
top5_genres=['Drama','Documnetary','Comedy','Comedy,Drama','Horror']
df=data.loc[:,['IMDb','Year','Genres']]
df['AverageRating']=df.groupby(['Year','Genres'])['IMDb'].transform('mean')
df=df[(df['Year']>2010) & (df['Year']<=2020)]
df=df.loc[df['Genres'].isin(top5_genres)]
df.drop('IMDb',axis=1,inplace=True)
df.sort_values('Year')
fig=px.bar(df,x='Genres', y='AverageRating', animation_frame='Year', 
           animation_group='Genres', color='Genres', hover_name='Genres', range_y=[0,10])
fig.update_layout(showlegend=False)
fig.show()
top5_netflix=data.loc[:,['Netflix','IMDb','Title','Year']]
top5_netflix=top5_netflix[top5_netflix['Netflix']==1]
top5_netflix=top5_netflix.sort_values('IMDb',ascending=False).head(5)
fig=px.bar(top5_netflix,x='Title',y='IMDb',height=600,color='IMDb',range_y=[0,10])
fig.update_layout(title='Top 5 Netflix Movies based on IMDb Rating')
fig.show()
top5_hulu=data.loc[:,['Hulu','IMDb','Title','Year']]
top5_hulu=top5_hulu[top5_hulu['Hulu']==1]
top5_hulu=top5_hulu.sort_values('IMDb',ascending=False).head(5)
fig=px.bar(top5_hulu,x='Title',y='IMDb',height=600,color='Title',range_y=[0,10])
fig.update_layout(title='Top 5 Hulu Movies based on IMDb Rating')
fig.show()

top5_prime=data.loc[:,['Prime Video','IMDb','Title','Year']]
top5_prime=top5_prime[top5_prime['Prime Video']==1]
top5_prime=top5_prime.sort_values('IMDb',ascending=False).head(5)
fig=px.bar(top5_prime,x='Title',y='IMDb',height=600,color='Title',range_y=[0,10])
fig.update_layout(title='Top 5 Prime Movies based on IMDb Rating')
fig.show()
top5_disney=data.loc[:,['Disney+','IMDb','Title','Year']]
top5_disney=top5_disney[top5_disney['Disney+']==1]
top5_disney=top5_disney.sort_values('IMDb',ascending=False).head(5)
fig=px.bar(top5_disney,x='Title',y='IMDb',height=600,color='Title',range_y=[0,10])
fig.update_layout(title='Top 5 Disney Movies based on IMDb Rating')
fig.show()
#top20_dir=data.groupby(['Directors','IMDb'])['Title'].count().reset_index().sort_values('IMDb',ascending=False).head(20)
top20_dir=data.loc[:,['Directors','IMDb','Title','Year']].sort_values('IMDb',ascending=False)

top20_dir.head(20)