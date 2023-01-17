import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))
def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")
movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")
credits.head()
movies.head()
movies.rename(columns={"id": "movie_id"}, inplace=True)
credits["cast"].iloc[0]
credits["crew"].iloc[0]
credits.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['cast']], axis=1)
credits.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['crew']], axis=1)
credits.apply(lambda row: [person.update({'order': order}) for order, person in enumerate(row['crew'])], axis=1)

cast_list = []
credits["cast"].apply(lambda x: cast_list.extend(x))
cast = pd.DataFrame(cast_list)
cast["type"] = "cast"

crew_list = []
credits["crew"].apply(lambda x: crew_list.extend(x))
crew = pd.DataFrame(crew_list)
crew["type"] = "crew"

people = pd.concat([cast, crew], ignore_index=True, sort=True)
del credits
people.head()
movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['genres']], axis=1)
movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['keywords']], axis=1)
movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['production_companies']], axis=1)
movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['production_countries']], axis=1)
movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['spoken_languages']], axis=1)

genres = []
movies["genres"].apply(lambda x: genres.extend(x))
genres = pd.get_dummies(pd.DataFrame(genres).drop("id", axis=1).set_index("movie_id")).sum(level=0)
genres.rename(columns = lambda x: str(x)[5:], inplace=True)

keywords = []
movies["keywords"].apply(lambda x: keywords.extend(x))
keywords = pd.get_dummies(pd.DataFrame(keywords).drop("id", axis=1).set_index("movie_id")).sum(level=0)
keywords.rename(columns = lambda x: str(x)[5:], inplace=True)

production_companies = []
movies["production_companies"].apply(lambda x: production_companies.extend(x))
production_companies = pd.get_dummies(pd.DataFrame(production_companies).drop("id", axis=1).set_index("movie_id")).sum(level=0)
production_companies.rename(columns = lambda x: str(x)[5:], inplace=True)

production_countries = []
movies["production_countries"].apply(lambda x: production_countries.extend(x))
production_countries = pd.get_dummies(pd.DataFrame(production_countries).drop("iso_3166_1", axis=1).set_index("movie_id")).sum(level=0)
production_countries.rename(columns = lambda x: str(x)[5:], inplace=True)

spoken_languages = []
movies["spoken_languages"].apply(lambda x: spoken_languages.extend(x))
spoken_languages = pd.get_dummies(pd.DataFrame(spoken_languages).drop("iso_639_1", axis=1).set_index("movie_id")).sum(level=0)
spoken_languages.rename(columns = lambda x: str(x)[5:], inplace=True)

movies.drop(["genres", "keywords", "production_companies", "production_countries", "spoken_languages"], axis=1, inplace=True)
movies.head()
movies.drop(["homepage", "original_title", "overview", "tagline"], axis=1, inplace=True)
sns.countplot(movies["status"])
movies.drop(["status"], axis=1, inplace=True)
movies["original_language"].value_counts().sort_values().tail().plot.barh()
movies.drop(["original_language"], axis=1, inplace=True)
movies["release_date"] = pd.to_datetime(movies["release_date"])
movies["release_year"] = movies["release_date"].dt.year
movies["release_month"] = movies["release_date"].dt.month
movies.drop(["release_date"], axis=1, inplace=True)
movies.set_index("movie_id", inplace=True)
movies.head()
fig, axarr = plt.subplots(4, 2, figsize=(24, 8))
sns.kdeplot(movies["budget"], ax=axarr[0][0])
axarr[0][0].xaxis.set_ticks(np.arange(0, 4.25e8, 0.25e8))
sns.kdeplot(movies["revenue"], ax=axarr[0][1])
axarr[0][1].xaxis.set_ticks(np.arange(0, 3e9, 0.25e9))
sns.kdeplot(movies["runtime"], ax=axarr[1][0])
sns.kdeplot(movies["popularity"], ax=axarr[1][1])
axarr[1][1].xaxis.set_ticks(np.arange(0, 900, 50))
sns.kdeplot(movies["vote_average"], ax=axarr[2][0])
axarr[2][0].xaxis.set_ticks(np.arange(0, 11, 1))
sns.kdeplot(movies["vote_count"], ax=axarr[2][1])
sns.kdeplot(movies["release_year"], ax=axarr[3][0])
sns.countplot(movies["release_month"], ax=axarr[3][1])
fig.tight_layout()
sns.countplot(people["type"])
sns.countplot(people.drop_duplicates(["id"])["type"])
fig, axarr= plt.subplots(1, 3, figsize=(24, 4))
sns.countplot(people.drop_duplicates(["id"])[people["type"] == "cast"]["gender"], ax=axarr[0])
sns.countplot(people.drop_duplicates(["id"])[people["type"] == "crew"]["gender"], ax=axarr[1])
sns.countplot(people.drop_duplicates(["id"])["gender"], ax=axarr[2])
axarr[0].set_title("Cast")
axarr[1].set_title("Crew")
axarr[2].set_title("Overall")
for i in range(3):
    axarr[i].set_xticklabels(["Undefined", "Male", "Female"])
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 4))
sns.countplot(y=people["department"], ax=axarr[0])
people["job"].value_counts().head(10).plot.barh(ax=axarr[1])
axarr[1].set_ylabel("job")
fig.tight_layout()
people[people["department"] == "Actors"]
fig, axarr = plt.subplots(3, 2, figsize=(20, 8))
genres.sum().plot.barh(ax=axarr[0][0])
keywords.sum().sort_values().tail(10).plot.barh(ax=axarr[0][1])
production_companies.sum().sort_values().tail(10).plot.barh(ax=axarr[1][0])
production_countries.sum().sort_values().tail(10).plot.barh(ax=axarr[1][1])
spoken_languages.sum().sort_values().tail().plot.barh(ax=axarr[2][0])
axarr[0][0].set_ylabel("genre")
axarr[0][1].set_ylabel("keyword")
axarr[1][0].set_ylabel("production_company")
axarr[1][1].set_ylabel("production_country")
axarr[2][0].set_ylabel("spoken_language")
axarr[2][1].axis("off")
fig.tight_layout()
ax = movies.nlargest(10, "revenue").iloc[::-1].plot.barh(x="title", y="revenue", legend=False)
ax.set_xlabel("revenue")
ax.set_ylabel("film")
fig, axarr = plt.subplots(4, 2, figsize=(20, 24))
p_color = dict(color="C0")
l_color = dict(color="C1")
sns.regplot(x="budget", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[0][0])
sns.regplot(x="runtime", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[0][1])
sns.regplot(x="release_year", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[1][0])
sns.boxplot(x="release_month", y="revenue", data=movies, ax=axarr[1][1])
sns.regplot(x="popularity", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[2][0])
sns.regplot(x="vote_average", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[2][1])
sns.regplot(x="vote_count", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[3][0])
fig.tight_layout()
