import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import json
from IPython.display import HTML
from wordcloud import WordCloud
import sqlite3 as sq3

api_key  = "api_key"
discover_api = "https://api.themoviedb.org/3/discover/movie?"
query = "&language=en-EN&sort_by=popularity.desc&primary_release_date.gte=2020-04-01&primary_release_date.lte=2020-09-02&page={}"
json_list = []
n_pages = 300
for i in range(1,n_pages+1):
    url = discover_api+api_key+query.format(i)
    r = requests.get(url)
    if r.status_code != 200:
        continue
    else:
        data = r.json()
        json_list.append(data) 
df = pd.DataFrame(json_list)
df.head()
df_disc_list = []
for i in range(0, n_pages):
    df_disc_list.append(pd.DataFrame(df["results"][i]))
    
df_disc = pd.concat(df_disc_list)
movie_id = df_disc["id"]
movie_id.reset_index(drop = True, inplace =True)
movie_api = "https://api.themoviedb.org/3/movie/{}?"
json_mov_list = []
for i in movie_id:
    url = movie_api.format(i) + api_key
    r = requests.get(url)
    if r.status_code != 200:
        continue
    else:
        data_mov = r.json()
        json_mov_list.append(data_mov) 
res = pd.DataFrame(json_mov_list)


res.to_json("movies.json", orient = "records")
with open("movies.json") as f:
    data = json.load(f)
df_mov = pd.json_normalize(data, sep = "_")
df_mov.info()
pd.to_datetime(df_mov["release_date"])
df_mov.loc[df_mov["vote_count"] == 0, "vote_average"] = np.nan
df_mov["runtime"].replace(0, np.nan, inplace = True)
df_mov["revenue"].replace(0, np.nan, inplace = True)
df_mov["budget"].replace(0, np.nan, inplace = True)
(df_mov["overview"]=="").value_counts(dropna = False).head(20)
df_mov["overview"].replace("", np.nan, inplace = True)
df_mov.isna().sum()
df_mov.dropna(thresh = 13)
df_mov["belongs_to_collection_name"].value_counts().sum()
columns = ["id", "title", "tagline", "release_date", "genres", "belongs_to_collection_name", 
       "original_language", "budget", "revenue", "production_companies",
       "production_countries", "vote_count", "vote_average", "popularity", "runtime",
       "overview", "spoken_languages", "poster_path"]
df_mov = df_mov.loc[:, columns]
df.reset_index(drop = True, inplace =True)
base_poster_url = 'http://image.tmdb.org/t/p/w185/'
df_mov["poster_path"] = "<img src='" + base_poster_url + df_mov["poster_path"] + "' style='height:100px;'>"
HTML(df_mov.loc[df_mov["popularity"]>1000, ["id", "title", "poster_path", "popularity"]].set_index("id").sort_values("popularity", ascending = False).to_html(escape=False))
HTML(df_mov.loc[df_mov["budget"]>50000000, ["id", "title", "poster_path", "budget", "genres"]].set_index("id").sort_values("budget", ascending = False).to_html(escape=False))
title = df_mov["title"].dropna()
overview = df_mov["overview"].dropna()
tagline = df_mov["tagline"].dropna()
title_list = ' '.join(title)
overview_list = ' '.join(overview)
tagline_list = ' '.join(tagline)
title_wordcloud = WordCloud(background_color='white', height=2000, width=4000, max_words= 200).generate(title_list)
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud, interpolation= "bilinear")
plt.axis('off')
plt.show()
tagline_wordcloud = WordCloud(background_color='white', height=2000, width=4000).generate(tagline_list)
plt.figure(figsize=(16,8))
plt.imshow(tagline_wordcloud, interpolation= "bilinear")
plt.axis('off')
plt.show()
overview_wordcloud = WordCloud(background_color='white', height=2000, width=4000).generate(overview_list)
plt.figure(figsize=(16,12))
plt.imshow(overview_wordcloud, interpolation= "bilinear")
plt.axis('off')
plt.show()
movies = df_mov[["id", "title", "revenue", "budget", "belongs_to_collection_name", "release_date", "vote_count", "vote_average"]].copy()
movies
genres = pd.json_normalize(data = data, record_path = "genres", meta = "id", record_prefix = "genre_")
genres
prod_comp = pd.json_normalize(data = data, record_path = "production_companies", meta = "id", record_prefix = "comp_")
prod_comp
country = pd.json_normalize(data = data, record_path = "production_companies", meta = "id", record_prefix = "comp_")
country
con = sq3.connect("movies.db")
movies.to_sql("Movies", con, index = False)
genres.to_sql("Genres", con, index = False)
prod_comp.to_sql("ProdCompanies", con, index = False)
country.to_sql("Country", con, index = False)
con.execute("Select * FROM sqlite_master").fetchall()
pd.read_sql("SELECT DISTINCT belongs_to_collection_name FROM Movies", con)
pd.read_sql("SELECT * FROM Movies ORDER BY budget DESC", con)
pd.read_sql("SELECT Movies.id, Movies.title, Movies.budget, Movies.vote_average, Genres.genre_name \
            FROM Movies \
            JOIN Genres \
            ON Movies.id=Genres.id \
            WHERE Movies.vote_average > 7 AND Movies.vote_count > 10 AND Movies.budget IS NOT NULL\
            ORDER BY Movies.budget DESC", con, index_col = "id")
pd.read_sql("SELECT SUM(Movies.budget), Genres.genre_name \
            FROM Movies \
            JOIN Genres \
            ON Movies.id=Genres.id \
            GROUP BY Genres.genre_name \
            ORDER BY SUM(Movies.budget) DESC", con)
pd.read_sql("SELECT SUM(Movies.revenue), ProdCompanies.comp_name \
            FROM Movies \
            JOIN ProdCompanies \
            ON Movies.id=ProdCompanies.id \
            GROUP BY ProdCompanies.comp_name \
            ORDER BY SUM(Movies.revenue) DESC", con).head(20)
