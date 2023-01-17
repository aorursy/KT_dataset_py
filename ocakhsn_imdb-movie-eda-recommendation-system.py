# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import ast

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
movies = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

credits = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")



print("Movies shape {}".format(movies.shape))

print("Credits shape {}".format(credits.shape))
movies.head()
credits.head()
def parsing(column, data):  

  data[column] = data[column].fillna('[]').apply(ast.literal_eval) # make it a list

  data[column] = data[column].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []) #iterate over its elements
def get_director(column, data):

    data[column] = data[column].fillna('[]').apply(ast.literal_eval) # make it a list

    data[column] = data[column].apply(lambda x: [i['name']  for i in x if i['job'] == "Director"] if isinstance(x, list) else []) #iterate over its elements
parsing('genres', movies)

parsing('production_companies', movies)

parsing('production_countries', movies)

parsing('spoken_languages', movies)

parsing('keywords', movies)

parsing('cast', credits)

get_director('crew', credits)
movies.head()
credits.head()
credits.rename(columns={'movie_id': 'id', 'crew': 'Director'}, inplace=True)
df = pd.merge(movies, credits.drop(columns=['title']), on="id")
directors = []

for i in df.index:

    current = df.iloc[i]['Director']

    if(len(current) > 0):

        directors.append(current[0])

    else:

        directors.append(" ")
df['Director'] = directors

df.head()
df.release_date = pd.to_datetime(df.release_date,  errors='coerce')

df['Year'] = df.release_date.dt.year

df['month'] = df.release_date.dt.month

df['weekday'] = df.release_date.dt.weekday
df['profit'] = df['revenue'] - df['budget']

df['profit_rate'] = df['profit'] / df['budget']



df.head()
revenues = df.groupby('Year')['revenue'].sum()

budgets = df.groupby('Year')['budget'].sum()

profits = df.groupby('Year')['profit'].sum()



fig = go.Figure()

fig.add_trace(go.Scatter(x=revenues.index, y=revenues.values,

                    mode='lines',marker_color='blue',name='Revenues'))



fig.add_trace(go.Scatter(x=budgets.index, y=budgets.values,

                    mode='lines',marker_color='red',name='Budgets'))



fig.add_trace(go.Scatter(x=profits.index, y=profits.values,

                    mode='lines',marker_color='green',name='Profits'))



fig.update_layout(title_text='Money Statistics')

fig.add_annotation(

            x=2012,

            y=revenues.values.max(),

            text="Peak")



fig.show()
twenty_two = df[df['Year'] == 2012]

twenty_two.sort_values(by="profit", ascending=False)[['title', 'budget', 'revenue', 'profit', 'genres']].head(10)
twenty_four = df[df['Year'] == 2014]

twenty_four.sort_values(by="profit", ascending=False)[['title', 'budget', 'revenue', 'profit', 'genres']].head(10)
sub_data = df[(df['budget'] != 0) & (df['revenue'] != 0) & (df['budget'] > 10000000)]

sub_data.sort_values(by='profit_rate', ascending=False)[['title', 'budget', 'profit', 'genres', 'profit_rate', 'Year']].head(20)
fig = px.scatter(df, x="budget", y="revenue", trendline="ols", title="Relationship between Budget and Revenue")

fig.update_layout(xaxis_title="Budget", 

                 yaxis_title="Revenue")

fig.show()
px.histogram(df[df['vote_count'] < 5000],x='vote_count',title='Distribution of Vote Counts',color_discrete_sequence=['#7D3C98'])
fig = px.bar(df.sort_values(by="vote_count", ascending=False).iloc[:20][::-1], x="vote_count", y="title", orientation='h', title="Movies with most vote counts")

fig.show()
fig = px.histogram(df,x='vote_average',title='Distribution of Vote Averages',color_discrete_sequence=['#7D3C98'])

fig.update_layout(xaxis_title="Vote Average", 

                 yaxis_title="Numbers")



fig.show()
from scipy.stats import skew

skew(df['vote_average'])
subdata = df[df['vote_count'] > 250]

fig = px.bar(subdata.sort_values(by="vote_average", ascending=False).iloc[:20][::-1], x="vote_average", y="title", orientation='h', title="Movies with most vote counts",color_discrete_sequence=['#52BE80'])

fig.update_layout(yaxis_title="Movie Title",

                 xaxis_title="İmdb Score")

fig.show()
def get_most_10(data, column):

    temp_data = data.apply(lambda x: pd.Series(x[column], dtype='object'),axis=1).stack().reset_index(level=1, drop=True)

    a = temp_data.value_counts().reset_index()

    a.rename(columns={0: 'Value'})

    fig = px.bar(a.iloc[:10][::-1], x=0, y='index', orientation='h', title="Top 10 {}".format(column),color_discrete_sequence=['#707B7C '])

    fig.show()

    
get_most_10(df, 'genres')
get_most_10(df, 'cast')
get_most_10(df, 'keywords')
get_most_10(df, 'production_countries')
most_profit_movies = df.sort_values(by="profit", ascending=False).head(200)

get_most_10( most_profit_movies, 'genres')
get_most_10( most_profit_movies, 'keywords')
get_most_10( most_profit_movies, 'cast')
get_most_10( most_profit_movies, 'spoken_languages')
years = df.groupby('Year')['title'].count()

fig = go.Figure()

fig.add_trace(go.Scatter(x=years.index, y=years.values,

                    mode='lines+markers',marker_color='cyan',name='Number of Movies in Years'))

fig.update_layout(title_text='Number of Movies in Years', 

                  plot_bgcolor='white',

                 xaxis_title="Years",

                 yaxis_title="Number of Movies")

fig.show()
months = df.groupby('month')['title'].count()

months_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

fig = go.Figure()

fig.add_trace(go.Bar(y=months_names[::-1], x=months.values[::-1],name='Number of Movies in Months', orientation='h'))

fig.update_layout(title_text='Number of Movies in Months',

                 xaxis_title="Number of Movies",

                 yaxis_title="Months",

                 plot_bgcolor='white')

fig.show()
days = df.groupby('weekday')['title'].count()

days_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday' ]

fig = go.Figure()

fig.add_trace(go.Bar(y=days_names[::-1], x=days.values[::-1],name='Number of Movies in Days', orientation='h'))

fig.update_layout(title_text='Number of Movies in Days',

                 xaxis_title="Number of Movies", 

                 yaxis_title="Week Days",

                 plot_bgcolor='white')

fig.show()
C = df['vote_average'].mean()

m = df['vote_count'].quantile(0.75)

print("c is {}".format(C))

print("m is {}".format(m))
df['imdb_rating'] = (df['vote_average']*df['vote_count'] + C*m) / (df['vote_count'] + m)

df['imdb_rating']
df.sort_values(by="imdb_rating", ascending=False)[['title', 'imdb_rating']].reset_index(drop=True).iloc[:10]
plt.figure(figsize=(15, 10))

data = df.sort_values(by="imdb_rating", ascending=False)[['title', 'imdb_rating']][:20]

fig = go.Figure()

fig.add_trace(go.Bar(y=data['title'][::-1], x=data['imdb_rating'][::-1],name='Number of Movies in Months', orientation='h'))

fig.update_layout(title_text='Movies with Highest IMDB RATING',

                 xaxis_title="Movie Names",

                 yaxis_title="İmdb Rating",

                 plot_bgcolor='white')

fig.show()
def get_3_elements(column, data):

    x1 = []

    x2 = []

    x3 = []



    length = data.shape[0]



    for i in range(length):

        current = data.iloc[i][column]

        cur_len = len(current)

        if cur_len >= 3:

            x1.append(current[0])

            x2.append(current[1])

            x3.append(current[2])

        elif cur_len == 2:

            x1.append(current[0])

            x2.append(current[1])

            x3.append("")

        elif cur_len == 1:

            x1.append(current[0])

            x2.append("")

            x3.append("")

        else:

            x1.append("")

            x2.append("")

            x3.append("")

    

    return x1, x2, x3
genre1, genre2, genre3 = get_3_elements('genres', df)

df['genre1'] = genre1

df['genre2'] = genre2

df['genre3'] = genre3

df[['genres', 'genre1', 'genre2', 'genre3']].head()
actor1,actor2, actor3 = get_3_elements('cast', df)

df['actor1'] = actor1

df['actor2'] = actor2

df['actor3'] = actor3

df[['title', 'cast', 'actor1', 'actor2', 'actor3']].head()
directors = df.groupby('Director')['imdb_rating'].agg(['min', 'max', 'mean', 'count']).reset_index()

selected_directors = directors[directors['count'] > 3].sort_values(by="mean", ascending=False).head(20).Director.values

selected_directors
fig = px.box(df, x="genre1", y="imdb_rating")

fig.show()
selected = df[df['Director'].isin(selected_directors)]

fig = px.box(selected, x="Director", y="imdb_rating", title="IMDB RATINGS OF DIRECTORS")

fig.show()
def get_recommend_by_genre_and_language(genre, language="en"):

    sub_data = df[(df['original_language'] == language) & (df['genre1'] == genre)& (df['vote_count'] > 250)]

    return sub_data.sort_values(by="imdb_rating",ascending=False)[['title', 'genres', 'vote_count', 'imdb_rating', 'original_language']].head(10)
get_recommend_by_genre_and_language('Drama', 'en')
recommend_columns = ['id', 'title', 'actor1', 'actor2', 'actor3', 'Director', 'genre1', 'genre2', 'genre3', 'imdb_rating']

movies_filter = df[recommend_columns]





def recommend_similar(movie):

    director_movie = movies_filter['Director'][movies_filter['id'] == movie].values[0]

    actor1_movie = movies_filter['actor1'][movies_filter['id'] == movie].values[0]

    actor2_movie = movies_filter['actor2'][movies_filter['id'] == movie].values[0]

    actor3_movie = movies_filter['actor3'][movies_filter['id'] == movie].values[0]

    genre1_movie = movies_filter['genre1'][movies_filter['id'] == movie].values[0]

    genre2_movie = movies_filter['genre2'][movies_filter['id'] == movie].values[0]

    genre3_movie = movies_filter['genre3'][movies_filter['id'] == movie].values[0]





    

    temp = movies_filter.copy()

    

    

    temp['same_director'] = np.nan

    temp['same_ac1'] = np.nan

    temp['same_ac2'] = np.nan

    temp['same_ac3'] = np.nan

    temp['same_g1'] = np.nan

    temp['same_g2'] = np.nan

    temp['same_g3'] = np.nan

    

    

    temp['same_director'] = (temp['Director'] ==  director_movie)

    temp['same_ac1'] = (temp['actor1'] ==  actor1_movie) | (temp['actor2'] ==  actor1_movie) | (temp['actor3'] ==  actor1_movie)

    temp['same_ac2'] = (temp['actor1'] ==  actor2_movie) | (temp['actor2'] ==  actor2_movie) | (temp['actor3'] ==  actor2_movie)

    temp['same_ac3'] = (temp['actor1'] ==  actor3_movie) | (temp['actor2'] ==  actor3_movie) | (temp['actor3'] ==  actor3_movie)

    temp['same_g1'] = (temp['genre1'] ==  genre1_movie) | (temp['genre2'] ==  genre1_movie) | (temp['genre3'] ==  genre1_movie)

    temp['same_g2'] = (temp['genre1'] ==  genre2_movie) | (temp['genre2'] ==  genre2_movie)  | (temp['genre3'] ==  genre2_movie)

    temp['same_g3'] = (temp['genre1'] ==  genre3_movie) | (temp['genre2'] ==  genre3_movie) | (temp['genre3'] ==  genre3_movie)

    

    temp['similar_count'] = temp['same_director'].astype(int) + temp['same_ac1'].astype(int) + temp['same_ac2'].astype(int) + temp['same_ac3'].astype(int) + temp['same_g1'].astype(int) + temp['same_g2'].astype(int) + temp['same_g3'].astype(int)

    

    result = temp.sort_values(by=["similar_count", 'imdb_rating'], ascending=False)[['id', 'title', 'similar_count', 'actor1', 'actor2', 'actor3','Director', 'genre1', 'genre2', 'genre3', 'imdb_rating']].head(10)

    

    return result



    
id_of_movie = df[df['title'] == "Eternal Sunshine of the Spotless Mind"]['id'].values[0]

a = recommend_similar(id_of_movie)

a
id_of_movie = df[df['title'] == "Titanic"]['id'].values[0]

a = recommend_similar(id_of_movie)

a
df[['title', 'overview']].head(5)
from sklearn.feature_extraction.text import TfidfVectorizer



df['overview'] = df['overview'].fillna('')



vectorizer = TfidfVectorizer(stop_words="english")





tf_idf_mat = vectorizer.fit_transform(df['overview'])



print("Shape is {}".format(tf_idf_mat.shape))
from sklearn.metrics.pairwise import linear_kernel



cosine_sim = linear_kernel(tf_idf_mat, tf_idf_mat)
cosine_sim[1].mean()
titles = pd.Series(df.index, index=df['title']).drop_duplicates()

titles
def get_recommendations(movie_title, cosine_similarity = cosine_sim):

    index_movie = titles[movie_title]

    

    similarities = cosine_similarity[index_movie]

    

    similarity_scores = list(enumerate(similarities))

    

    similarity_scores = sorted(similarity_scores , key=lambda x: x[1], reverse = True)

    

    similarity_scores = similarity_scores[1:11]

    

    similar_indexes = [x[0] for x in similarity_scores]

    

    print("İf you liked {}, I can recommend you these movies based on the overview\n\n".format(movie_title))

    

    return df['title'].iloc[similar_indexes]

    

get_recommendations('The Dark Knight')
keyword1, keyword2, keyword3 = get_3_elements('keywords', df)



df['keyword1'] = keyword1

df['keyword2'] = keyword2

df['keyword3'] = keyword3



df[['title', 'keyword1', 'keyword2', 'keyword3']].head(5)
df2 = df.copy()



df2['total'] = ""

columns_to_be_added = ['keyword1', 'keyword2', 'keyword3', 'actor1', 'actor2', 'actor3', 'genre1', 'genre2', 'genre3', 'Director']



for c in columns_to_be_added:

    df2['total'] += (df2[c].str.lower() + " ")



df2['total']
vectorizer2 = TfidfVectorizer(stop_words="english")



X2 = vectorizer2.fit_transform(df2['total'])



print("Shape is {}".format(X2.shape))
cosine_sim2 = linear_kernel(X2, X2)

cosine_sim2.shape
get_recommendations('The Dark Knight', cosine_sim2)
get_recommendations('The Godfather', cosine_sim2)