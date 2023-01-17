import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline

from wordcloud import WordCloud, ImageColorGenerator



data = pd.read_csv('../input/netflix-shows/netflix_titles.csv')

ratings = pd.read_csv('../input/imdb-ratings-for-the-netflix-shows/IMDB_results_jan-28-2020.csv')

data = data.fillna('Unknown')

data.drop(['show_id','date_added'], axis=1, inplace=True)
text = " ".join(review for review in data.description)

wordcloud = WordCloud(max_words=200, colormap='Set3',background_color="black").generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
df = data[data['release_year']>2010]

sns.kdeplot(data=df['release_year'], label='release_year', shade=True)

plt.title('Number of shows released per year')

plt.show()
data['IMDB_rating'] = ratings['IMDB_rating'].str.split().str[0].str.replace('Not','0.0')

data['Number of reviews'] = ratings['IMDB_rating'].str.split().str[3].str.replace(',','')

data['IMDB_rating'] = data['IMDB_rating'].astype('float')

data['Number of reviews'] = data['Number of reviews'].dropna().astype('int')

data = data.dropna()
df = data.sort_values('Number of reviews', ascending=False)

df = df.head()



fig = plt.figure(figsize=(10,7))

plt.pie(df['Number of reviews'], labels=df['title'], autopct='%1.1f%%', shadow=True)

centre_circle = plt.Circle((0,0),0.45,color='black', fc='white',linewidth=1.25)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')

plt.show()
sns.kdeplot(data=data['IMDB_rating'], label='IMDB_rating', shade=True)

plt.title('Ratings of the Netflix shows and movies')

plt.show()
movie = data[data['type']=='Movie']

movie = movie.sort_values('IMDB_rating')

movie = movie.tail()



fig = px.pie(movie, names='title', values='IMDB_rating', template='seaborn')

fig.update_traces(rotation=90, pull=[0.03,0.03,0.03,0.03,0.2], textinfo="percent+label")

fig.show()
show = data[data['type']=='TV Show']

show = show.sort_values('IMDB_rating')

show = show.tail()



fig = px.pie(show, names='title', values='IMDB_rating', template='seaborn')

fig.update_traces(rotation=90, pull=[0.03,0.03,0.03,0.03,0.2], textinfo="percent+label")

fig.show()
top_5_genres = ['Stand-Up Comedy','Documentaries','Dramas, International Movies','Comedies, Dramas, International Movies',"Kids' TV"]

perc = data.loc[:,["release_year","listed_in",'IMDB_rating']]

perc['mean_rating'] = perc.groupby([perc.listed_in,perc.release_year])['IMDB_rating'].transform('mean')

perc.drop('IMDB_rating', axis=1, inplace=True)

perc = perc.drop_duplicates()

perc = perc[(perc.release_year>2009) & (perc.release_year<2020)]

perc = perc.loc[perc['listed_in'].isin(top_5_genres)]

perc = perc.sort_values("release_year")



fig=px.bar(perc,x='listed_in', y="mean_rating", animation_frame="release_year", 

           animation_group="listed_in", color="listed_in", hover_name="listed_in", range_y=[0,10])

fig.update_layout(showlegend=False)

fig.show()
rate = data.loc[:,['rating','Number of reviews']]

rate['mean_num_of_reviews'] = rate.groupby('rating')['Number of reviews'].transform('mean')

rate.drop('Number of reviews', axis=1, inplace=True)

rate = rate.drop_duplicates().sort_values('mean_num_of_reviews')

rate = rate.tail()



fig = px.pie(rate, names='rating', values='mean_num_of_reviews', template='seaborn')

fig.update_traces(rotation=90, pull=[0.03,0.03,0.03,0.03,0.2], textinfo="percent+label")

fig.show()
ax = sns.stripplot(x="type", y="IMDB_rating", data=data)

ax = sns.violinplot(x="type", y="IMDB_rating", data=data)
directors = data[data['director']!='Unknown']

directors = directors.sort_values('IMDB_rating', ascending=False)

directors = directors.head()



text = ",".join(review for review in directors.director)

wordcloud = WordCloud(max_words=200, colormap='Set3',background_color="black").generate(text)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
movie = data.loc[data['type']=='Movie', ['duration','IMDB_rating']]

movie = movie.sort_values('IMDB_rating', ascending=False)

movie = movie.head()



plt.bar(x=movie['duration'], height=movie['IMDB_rating'])

plt.ylim(8.5, 10)

plt.show()
show = data.loc[data['type']=='TV Show', ['duration','IMDB_rating']]

show['IMDB_rating'] = show.groupby('duration')['IMDB_rating'].transform('max')

show = show.drop_duplicates()

show = show.sort_values('IMDB_rating', ascending=False)

show = show.head(5)



fig = px.pie(show, names='duration', values='IMDB_rating', template='seaborn')

fig.update_traces(rotation=90, pull=[0.1,0.03,0.03,0.03,0.03], textinfo="percent+label")

fig.show()
df = data[data['release_year']>2010]

sns.kdeplot(data=df['Number of reviews'], label='Number of reviews', shade=True)

plt.title('Distribution of the Number of reviews')

plt.show()
country = data.loc[data['type']=='Movie', ['country','IMDB_rating']]

country['IMDB_rating'] = country.groupby('country')['IMDB_rating'].transform('max')

country = country.drop_duplicates()

country = country.sort_values('IMDB_rating', ascending=False)

country = country[1:6]



fig = px.pie(country, names='country', values='IMDB_rating', template='seaborn')

fig.update_traces(rotation=90, pull=[0.1,0.03,0.03,0.03,0.03], textinfo="percent+label")

fig.show()
movie = data[data['type']=='Movie']

movie = movie.sort_values('IMDB_rating')

movie = movie.tail(1)

text = 'ShahRukhKhan DavidLetterman'

wordcloud = WordCloud(colormap='Set3',background_color="black").generate(text)

plt.figure(figsize=(10,7))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()