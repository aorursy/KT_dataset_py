# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

  #  for filename in filenames:

   #     print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

netflix_overall=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

netflix_overall.head()
netflix_overall.count()
netflix_shows=netflix_overall[netflix_overall['type']=='TV Show']

netflix_movies=netflix_overall[netflix_overall['type']=='Movie']
sns.set(style="darkgrid")

ax = sns.countplot(x="type", data=netflix_overall, palette="Set2")
netflix_date = netflix_shows[['date_added']].dropna()

netflix_date['year'] = netflix_date['date_added'].apply(lambda x : x.split(', ')[-1])

netflix_date['month'] = netflix_date['date_added'].apply(lambda x : x.lstrip().split(' ')[0])



month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][::-1]

df = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)[month_order].T

plt.figure(figsize=(10, 7), dpi=200)

plt.pcolor(df, cmap='afmhot_r', edgecolors='white', linewidths=2) # heatmap

plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, fontsize=7, fontfamily='serif')

plt.yticks(np.arange(0.5, len(df.index), 1), df.index, fontsize=7, fontfamily='serif')



plt.title('Netflix Contents Update', fontsize=12, fontfamily='calibri', fontweight='bold', position=(0.20, 1.0+0.02))

cbar = plt.colorbar()



cbar.ax.tick_params(labelsize=8) 

cbar.ax.minorticks_on()

plt.show()
plt.figure(figsize=(12,10))

sns.set(style="darkgrid")

ax = sns.countplot(x="rating", data=netflix_movies, palette="Set2", order=netflix_movies['rating'].value_counts().index[0:15])
imdb_ratings=pd.read_csv('/kaggle/input/imdb-extensive-dataset/IMDb ratings.csv',usecols=['weighted_average_vote'])

imdb_titles=pd.read_csv('/kaggle/input/imdb-extensive-dataset/IMDb movies.csv', usecols=['title','year','genre'])

ratings = pd.DataFrame({'Title':imdb_titles.title,

                    'Release Year':imdb_titles.year,

                    'Rating': imdb_ratings.weighted_average_vote,

                    'Genre':imdb_titles.genre})

ratings.drop_duplicates(subset=['Title','Release Year','Rating'], inplace=True)

ratings.shape
ratings.dropna()

joint_data=ratings.merge(netflix_overall,left_on='Title',right_on='title',how='inner')

joint_data=joint_data.sort_values(by='Rating', ascending=False)
import plotly.express as px

top_rated=joint_data[0:10]

fig =px.sunburst(

    top_rated,

    path=['title','country'],

    values='Rating',

    color='Rating')

fig.show()
country_count=joint_data['country'].value_counts().sort_values(ascending=False)

country_count=pd.DataFrame(country_count)

topcountries=country_count[0:11]

topcountries
import plotly.express as px

data = dict(

    number=[1063,619,135,60,44,41,40,40,38,35],

    country=["United States", "India", "United Kingdom", "Canada", "Spain",'Turkey','Philippines','France','South Korea','Australia'])

fig = px.funnel(data, x='number', y='country')

fig.show()
plt.figure(figsize=(12,10))

sns.set(style="darkgrid")

ax = sns.countplot(y="release_year", data=netflix_movies, palette="Set2", order=netflix_movies['release_year'].value_counts().index[0:15])
countries={}

netflix_movies['country']=netflix_movies['country'].fillna('Unknown')

cou=list(netflix_movies['country'])

for i in cou:

    #print(i)

    i=list(i.split(','))

    if len(i)==1:

        if i in list(countries.keys()):

            countries[i]+=1

        else:

            countries[i[0]]=1

    else:

        for j in i:

            if j in list(countries.keys()):

                countries[j]+=1

            else:

                countries[j]=1
countries_fin={}

for country,no in countries.items():

    country=country.replace(' ','')

    if country in list(countries_fin.keys()):

        countries_fin[country]+=no

    else:

        countries_fin[country]=no

        

countries_fin={k: v for k, v in sorted(countries_fin.items(), key=lambda item: item[1], reverse= True)}

plt.figure(figsize=(8,8))

ax = sns.barplot(x=list(countries_fin.keys())[0:10],y=list(countries_fin.values())[0:10])

ax.set_xticklabels(list(countries_fin.keys())[0:10],rotation = 90)

netflix_movies['duration']=netflix_movies['duration'].str.replace(' min','')

netflix_movies['duration']=netflix_movies['duration'].astype(str).astype(int)

netflix_movies['duration']
sns.set(style="darkgrid")

sns.kdeplot(data=netflix_movies['duration'], shade=True)
from collections import Counter



genres=list(netflix_movies['listed_in'])

gen=[]



for i in genres:

    i=list(i.split(','))

    for j in i:

        gen.append(j.replace(' ',""))

g=Counter(gen)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



text = list(set(gen))

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(str(text))



plt.imshow(wordcloud,interpolation="bilinear")

plt.axis("off")

plt.show()
g={k: v for k, v in sorted(g.items(), key=lambda item: item[1], reverse= True)}





fig, ax = plt.subplots()



fig = plt.figure(figsize = (10, 10))

x=list(g.keys())

y=list(g.values())

ax.vlines(x, ymin=0, ymax=y, color='green')

ax.plot(x,y, "o", color='maroon')

ax.set_xticklabels(x, rotation = 90)

ax.set_ylabel("Count of movies")

# set a title

ax.set_title("Genres");
countries1={}

netflix_shows['country']=netflix_shows['country'].fillna('Unknown')

cou1=list(netflix_shows['country'])

for i in cou1:

    #print(i)

    i=list(i.split(','))

    if len(i)==1:

        if i in list(countries1.keys()):

            countries1[i]+=1

        else:

            countries1[i[0]]=1

    else:

        for j in i:

            if j in list(countries1.keys()):

                countries1[j]+=1

            else:

                countries1[j]=1
countries_fin1={}

for country,no in countries1.items():

    country=country.replace(' ','')

    if country in list(countries_fin1.keys()):

        countries_fin1[country]+=no

    else:

        countries_fin1[country]=no

        

countries_fin1={k: v for k, v in sorted(countries_fin1.items(), key=lambda item: item[1], reverse= True)}

# Set the width and height of the figure

plt.figure(figsize=(15,15))



# Add title

plt.title("Content creating countries")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(y=list(countries_fin1.keys()), x=list(countries_fin1.values()))



# Add label for vertical axis

plt.ylabel("Arrival delay (in minutes)")
features=['title','duration']

durations= netflix_shows[features]



durations['no_of_seasons']=durations['duration'].str.replace(' Season','')



#durations['no_of_seasons']=durations['no_of_seasons'].astype(str).astype(int)

durations['no_of_seasons']=durations['no_of_seasons'].str.replace('s','')
durations['no_of_seasons']=durations['no_of_seasons'].astype(str).astype(int)

t=['title','no_of_seasons']

top=durations[t]



top=top.sort_values(by='no_of_seasons', ascending=False)
top20=top[0:20]

top20.plot(kind='bar',x='title',y='no_of_seasons', color='red')
bottom=top.sort_values(by='no_of_seasons')

bottom=bottom[20:50]



import plotly.graph_objects as go



fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'No of seasons']),

                 cells=dict(values=[bottom['title'],bottom['no_of_seasons']],fill_color='lavender'))

                     ])

fig.show()
genres=list(netflix_shows['listed_in'])

gen=[]



for i in genres:

    i=list(i.split(','))

    for j in i:

        gen.append(j.replace(' ',""))

g=Counter(gen)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



text = list(set(gen))



wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate(str(text))

plt.rcParams['figure.figsize'] = (13, 13)

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis("off")

plt.show()
us_series_data=netflix_shows[netflix_shows['country']=='United States']
oldest_us_series=us_series_data.sort_values(by='release_year')[0:20]
fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'Release Year'],fill_color='paleturquoise'),

                 cells=dict(values=[oldest_us_series['title'],oldest_us_series['release_year']],fill_color='pink'))

                     ])

fig.show()
newest_us_series=us_series_data.sort_values(by='release_year', ascending=False)[0:50]
fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'Release Year'],fill_color='yellow'),

                 cells=dict(values=[newest_us_series['title'],newest_us_series['release_year']],fill_color='lavender'))

                     ])

fig.show()
netflix_fr=netflix_overall[netflix_overall['country']=='France']

nannef=netflix_fr.dropna()

import plotly.express as px

fig = px.treemap(nannef, path=['country','director'],

                  color='director', hover_data=['director','title'],color_continuous_scale='Purples')

fig.show()

newest_fr_series=netflix_fr.sort_values(by='release_year', ascending=False)[0:20]
newest_fr_series
fig = go.Figure(data=[go.Table(header=dict(values=['Title', 'Release Year']),

                 cells=dict(values=[newest_fr_series['title'],newest_fr_series['release_year']]))

                     ])

fig.show()
topdirs=pd.value_counts(netflix_overall['duration'])

fig = go.Figure([go.Bar(x=topdirs.index, y=topdirs.values , text=topdirs.values,marker_color='indianred')])

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()
from sklearn.feature_extraction.text import TfidfVectorizer
#removing stopwords

tfidf = TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

netflix_overall['description'] = netflix_overall['description'].fillna('')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(netflix_overall['description'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape

# Import linear_kernel

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(netflix_overall.index, index=netflix_overall['title']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):

    idx = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return netflix_overall['title'].iloc[movie_indices]
get_recommendations('Peaky Blinders')
get_recommendations('Mortel')
filledna=netflix_overall.fillna('')

filledna.head(2)
def clean_data(x):

        return str.lower(x.replace(" ", ""))
features=['title','director','cast','listed_in','description']

filledna=filledna[features]
for feature in features:

    filledna[feature] = filledna[feature].apply(clean_data)

    

filledna.head(2)
def create_soup(x):

    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']
filledna['soup'] = filledna.apply(create_soup, axis=1)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(filledna['soup'])



cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
filledna=filledna.reset_index()

indices = pd.Series(filledna.index, index=filledna['title'])
def get_recommendations_new(title, cosine_sim=cosine_sim):

    title=title.replace(' ','').lower()

    idx = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return netflix_overall['title'].iloc[movie_indices]
get_recommendations_new('PK', cosine_sim2)
get_recommendations_new('Peaky Blinders', cosine_sim2)
get_recommendations_new('The Hook Up Plan', cosine_sim2)
books=pd.read_csv('/kaggle/input/goodbooks-10k/books.csv')

books['original_title']=books['original_title'].str.lower()

x=netflix_overall

x['title']=x['title'].str.lower()

t=x.merge(books, left_on='title', right_on='original_title', how="inner")
t.shape
import plotly.graph_objects as go



labels = ['Shows from books','Shows not from books']

values = [248,6234]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()