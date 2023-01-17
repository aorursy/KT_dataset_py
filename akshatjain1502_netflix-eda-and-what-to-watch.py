import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
%matplotlib inline
import math as math
import time 
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [14,14]
import networkx as nx

import warnings
warnings.filterwarnings("ignore")
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
df = pd.read_csv(r"/kaggle/input/netflix-shows/netflix_titles.csv");
df.head()
import nltk
import pandas_profiling
df.profile_report(title='Netflix Reviews - Report' , progress_bar = False)
df.shape
df.info()
df.corr()
df.isnull().sum()
sns.heatmap(df.isnull(), cmap = 'viridis')
df["rating"].value_counts()
df.drop(["date_added", "cast"], inplace = True, axis = 1)
df["country"].replace(np.nan, 'United States',inplace =  True)
df["rating"].replace(np.nan, 'TV-MA',inplace =  True)
df.head()
df["listed_in"].value_counts()
df["type"].value_counts()
plt.figure(figsize = (12, 8))
sns.countplot(data = df, x = "type")
plt.figure(figsize = (12, 8))
sns.countplot(data = df, x = "rating")
plt.figure(figsize = (36, 8))
sns.countplot(data = df, x = "release_year")
plt.figure(figsize = (16, 6))
sns.scatterplot(data = df, x = "rating", y = "type")
plt.figure(figsize = (16, 6))
sns.countplot(data = df, x = "rating", hue = "type")
import plotly.express as px #distribution according to countries
top_rated=df[0:10]
fig =px.sunburst(top_rated,path=['country'])
fig.show()
df["rating"].value_counts().plot.pie(autopct = "%1.1f%%", figsize = (20,35))
plt.show()
counter_country = df["country"].value_counts().sort_values(ascending= False)

counter_country = pd.DataFrame(counter_country)
topcountry = counter_country[0:11]
topcountry
old = df.sort_values("release_year", ascending= True)
old = old[old["duration"] != ""]
old[['title', "release_year"]][:10]
tag = "Stand-Up Comedy" #standup shows on Netflix
df["relevant"] = df['listed_in'].fillna("").apply(lambda x : 1 if tag.lower() in x.lower() else 0)
com = df[df["relevant"] == 1]
com[com["country"] == "United States"][["title", "country","release_year"]].head(10)
tag = "Kids' TV" #Kids' TV shows on Netflix
df["relevant"] = df['listed_in'].fillna("").apply(lambda x : 1 if tag.lower() in x.lower() else 0)
com = df[df["relevant"] == 1]
com[com["country"] == "United States"][["title", "country","release_year"]].head(10)
df_countries = pd.DataFrame(df.country.value_counts().reset_index().values,  columns= ["country", "count"])
df_countries.head(10)
fig = px.choropleth(locationmode="country names", locations= df_countries.country, labels=df_countries["count"], 
                    hover_name=df_countries["country"])
fig.show()
date = pd.DataFrame(df.release_year.value_counts().reset_index().values,  columns= ["year", "count"])
date.head(10)
plt.figure(figsize=(12,6))
df[df["type"] == "Movie"]["release_year"].value_counts()[:20].plot(kind = "bar", color = "blue")
plt.title("Frequency of Movies which were released in different years and are available on Netflix")
plt.figure(figsize=(12,6))
df[df["type"] == "TV Show"]["release_year"].value_counts()[:20].plot(kind = "bar", color = "red")
plt.title("Frequency of TV Show which were released in different years and are available on Netflix")
plt.figure(figsize=(12,6))
df[df["type"] == "Movie"]["listed_in"].value_counts()[:10].plot(kind = "barh", color = "black")
plt.title("Top 10 Genres of Movies",size=18)
plt.figure(figsize=(12,6))
df[df["type"] == "TV Show"]["listed_in"].value_counts()[:10].plot(kind = "barh", color = "black")
plt.title("Top 10 Genres of TV Show",size=18)
from wordcloud import WordCloud

plt.subplots(figsize=(25,15))
wordcloud = WordCloud(background_color='Black',width=1920,height=1080).generate(" ".join(df.title))

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('cast.png')
plt.show()
df = df[["title","director", "listed_in", "description"]]
df.head()
df.director.fillna("", inplace = True)
df["movie_info"] = df["director"] + ' ' + df["listed_in"] + ' ' + df["description"]
df.head()
df = df[["title", 'movie_info']]
df.head()
from nltk.corpus import stopwords
#nltk.download()
import string
stop = set(stopwords.words("english"))
punctuation = list(string.punctuation)
stop.update(punctuation)
from nltk.stem import WordNetLemmatizer
import nltk
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    final_text = []
    for i in text.split():
         if i.strip().lower() not in stop:
                word = lemmatizer.lemmatize(i.strip())
                final_text.append(word.lower())
                
    return  " ".join(final_text)      
                
df.movie_info = df.movie_info.apply(lemmatize_words)
df.head()
from sklearn.feature_extraction.text import CountVectorizer
tf = CountVectorizer()
X=tf.fit_transform(df["movie_info"])
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(X)
liked_movie = 'Paap-O-Meter'
index_l = df[df['title'] == liked_movie].index.values[0]
similar_movies = list(enumerate(cosine_sim[index_l]))
sort_movies = sorted(similar_movies , key = lambda X:X[1] , reverse = True)
sort_movies.pop(0)
sort_movies = sort_movies[:10]
sort_movies
for movies in sort_movies:
    print(df.title[movies[0]])


