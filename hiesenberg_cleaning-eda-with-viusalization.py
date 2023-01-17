# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
import plotly
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected = True)
import plotly.graph_objs as go
from collections import Counter,OrderedDict
sns.set()
df=pd.read_csv('../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()
df.shape
df.isnull().sum()
df.drop(['Age','Rotten Tomatoes','Type'],axis=1,inplace=True)
df=df.dropna()
df.isnull().sum()
df=df.reset_index()
df.drop('index',axis=1,inplace=True)
df['Service Provider'] = df.loc[:,['Netflix','Prime Video','Disney+','Hulu']].idxmax(axis=1)
df.drop(['Netflix','Prime Video','Disney+','Hulu'],axis=1,inplace=True)
#Distribution of OTT Platforms 
ott_platform =  ['Netflix', 'Prime Video', 'Disney+', 'Hulu']
sizes = [len(df[df['Service Provider']=='Netflix']), len(df[df['Service Provider']=='Prime Video']), len(df[df['Service Provider']=='Disney+']), len(df[df['Service Provider']=='Hulu'])]
trace = go.Pie(labels = ott_platform, values = sizes)
data = [trace]
fig = go.Figure(data = data)
iplot(fig)
#Distribution of movie rating in each platform
plt.figure(figsize=(15,8))
sns.violinplot(x=df['IMDb'],y=df['Service Provider'])
#Highest Rated Movie Each Year
movie_each_year= df.loc[df.groupby("Year")["IMDb"].idxmax()].reset_index()
movie_each_year.drop('index',axis=1,inplace=True)
movie_each_year.loc[:,['Title','Year','IMDb']]
#Change of best movie rating over the years
plt.figure(figsize=(10,5))
sns.lineplot(x='IMDb',y='Year',data=movie_each_year)
#Effect of runtime on rating
plt.figure(figsize=(10,5))
sns.lineplot(x='Runtime',y='IMDb',data=df)
#Seperating based on ott platform
netflix=[]
prime=[]
disney=[]
hulu=[]

for i in range(len(df)):
    if(df['Service Provider'][i]=='Netflix'):
        netflix.append(df['Genres'][i])
    elif(df['Service Provider'][i]=='Prime Video'):
        prime.append(df['Genres'][i])
    elif(df['Service Provider'][i]=='Disney+'):
        disney.append(df['Genres'][i])
    elif(df['Service Provider'][i]=='Hulu'):
        hulu.append(df['Genres'][i])

netflix=" ".join(netflix)
prime=" ".join(prime)
disney=" ".join(disney)
hulu=" ".join(hulu)
#Most common genres of movies in Netflix
wordcloud = WordCloud(stopwords=STOPWORDS).generate(netflix) 
  
# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#Most common genres of movies in Prime
wordcloud = WordCloud(stopwords=STOPWORDS).generate(prime) 
  
# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#Most common genres of movies in Disney+
wordcloud = WordCloud(stopwords=STOPWORDS).generate(disney) 
  
# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#Most common genres of movies in Hulu
wordcloud = WordCloud(stopwords=STOPWORDS).generate(hulu) 
  
# plot the WordCloud image
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#Average Runtime length of each ott platform
df.groupby("Service Provider")["Runtime"].mean()
#Average IMDb rating of each ott platform
df.groupby("Service Provider")["IMDb"].mean()
#Runtime length movie changing in years
plt.figure(figsize=(10,5))
sns.lineplot(x='Year',y='Runtime',data=df)
#Top 10 movies
df.sort_values('IMDb',ascending=False)[:10].reset_index().drop(['ID','index','Country','Language'],axis=1)
#Top 10 movies of netflix
df[df['Service Provider']=='Netflix'].sort_values('IMDb',ascending=False)[:10].reset_index().drop(['ID','index','Country','Language'],axis=1)
#Top 10 movies of prime
df[df['Service Provider']=='Prime Video'].sort_values('IMDb',ascending=False)[:10].reset_index().drop(['ID','index','Country','Language'],axis=1)
#Top 10 movies of disney+
df[df['Service Provider']=='Disney+'].sort_values('IMDb',ascending=False)[:10].reset_index().drop(['ID','index','Country','Language'],axis=1)
#Top 10 movies of hulu
df[df['Service Provider']=='Hulu'].sort_values('IMDb',ascending=False)[:10].reset_index().drop(['ID','index','Country','Language'],axis=1)
#Collecting the genres
genres=[]
for i in range(len(df)):
    x=df['Genres'][i]
    y=x.split(',')
    for j in range(len(y)):
        genres.append(y[j])

genre_c = Counter(genres)
genre_c
#Most popular genres
data = [go.Bar(
   x = list(genre_c.keys()),
   y = list(genre_c.values())
)]
fig = go.Figure(data=data)
iplot(fig)
#Collecting the directors names
directors=[]
for i in range(len(df)):
    x=df['Directors'][i]
    y=x.split(',')
    for j in range(len(y)):
        directors.append(y[j])
        
director_c = Counter(directors)
director_c_10=OrderedDict(director_c.most_common()[:10])

#Directors with most movies
data = [go.Bar(
   x = list(director_c_10.keys()),
   y = list(director_c_10.values()))]
fig = go.Figure(data=data)
iplot(fig)
#Extracting the countries
country=[]
for i in range(len(df)):
    x=df['Country'][i]
    y=x.split(',')
    for j in range(len(y)):
        country.append(y[j])
        
country_c = Counter(country)
country_c_15=OrderedDict(country_c.most_common()[:15])

#Countries with most movies
data = [go.Bar(
   x = list(country_c_15.keys()),
   y = list(country_c_15.values())
)]
fig = go.Figure(data=data)
iplot(fig)
