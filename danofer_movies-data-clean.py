import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

from ast import literal_eval
import ast
PATH = "../input/"
df_credits = pd.read_csv(PATH+"credits.csv")
print("credits: \n ", df_credits.columns)
df_keywords = pd.read_csv(PATH+"keywords.csv")
print("\n keywords: \n", df_keywords.columns)

df_links = pd.read_csv(PATH+"links.csv")
print("\n links: \n", df_links.columns)
df = pd.read_csv(PATH+"movies_metadata.csv")
df.head()
df[["id","imdb_id"]].dtypes
df.isnull().sum()

# there's also 1 missing for cast/crew : we'll drop that row later
# Only 3 bad rows!
print (df[pd.to_numeric(df['id'], errors='coerce').isnull()])
df.shape
df["id"] =pd.to_numeric(df['id'], errors='coerce',downcast="integer")
# df["imdb_id"] =pd.to_numeric(df['imdb_id'], errors='coerce',downcast="integer")
df.dropna(subset=["id"],inplace=True)
df.shape
df.dropna(subset=["imdb_id"]).shape
# print(df.shape)
print("df_credits",df_credits.shape)

df_credits.head(3)
df.head(3)
df = df.merge(df_credits,on=["id"],how="left")
print(df_links.shape)
df_links.head()
df = df.merge(df_keywords,on=["id"],how="left")
print("missing imdb ids in main data:",df["imdb_id"].isnull().sum())
print("LINK data missing Imdb ids:",df_links["imdbId"].isnull().sum())
print("LINK data missing Tmdb ids:",df_links["tmdbId"].isnull().sum())
# looks like our coercion lost many rows. We don't really care about imdb IDs so we can ignore frankly..
# we see matching is poor for ID-movieId- but do we care? 
df.drop(["imdb_id"],axis=1).merge(df_links,left_on="id",right_on="movieId",how="inner").shape
# imdbID match rate: 
df.merge(df_links,left_on="imdb_id",right_on="imdbId",how="inner").shape
# we don't merge due to bad ,atching - data wrangling issue, can be fixed
# df = df.merge(df_links,left_on="id",right_on="movieId",how="left")
df.shape
df.head()
df.shape
df.drop_duplicates().shape
df.isnull().sum()
df.dropna(subset=["cast","crew","keywords","popularity"],inplace=True)
print(df.shape)
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['budget'] = df['budget'].replace(0, np.nan)

df['revenue'] = df['revenue'].replace(0, np.nan)
vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

m = vote_counts.quantile(0.75)

def weighted_rating(x):
    v = x['vote_count']+1 # added +1 - Dan
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

df['weighted_rating'] = df.apply(weighted_rating, axis=1)

df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['cast'] = df['cast'].apply(literal_eval)
df['crew'] = df['crew'].apply(literal_eval)
df['keywords'] = df['keywords'].apply(literal_eval)
df['cast_size'] = df['cast'].apply(lambda x: len(x))
df['crew_size'] = df['crew'].apply(lambda x: len(x))
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
df['director'] = df['crew'].apply(get_director)
# import json
# df['keywords'].iloc[0].replace("'",'"').str

# # works for 1 row:
# json.loads(df['keywords'].iloc[0].replace("'",'"'))
df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['cast'] = df['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

df['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['production_countries'].head()
df['production_countries'] = df['production_countries'].fillna('[]').apply(ast.literal_eval)
df['production_countries'] = df['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['production_countries'].head(2)
df['revenue_divide_budget'] = df['revenue'] / df['budget']
df['belongs_to_collection'].head()
# https://stackoverflow.com/questions/26614465/python-pandas-apply-function-if-a-column-value-is-not-null

# df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda row: ast.literal_eval(row) if (row.notnull().all()) , axis=1).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)

#  df['belongs_to_collection'].apply(lambda row: ast.literal_eval(row) if (row.notnull().all()) else row)

#  df['belongs_to_collection'].apply(lambda row: ast.literal_eval(row) if (row.isnull()== False) else row)
df['belongs_to_collection'] = df['belongs_to_collection'].fillna("[]").apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)

## https://www.kaggle.com/rounakbanik/the-story-of-film
df['spoken_languages'] = df['spoken_languages'].fillna('[]').apply(ast.literal_eval).apply(lambda x: len(x) if isinstance(x, list) else np.nan)
df.head(20)['production_companies'].fillna("[]").apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['production_companies'] = df['production_companies'].apply(ast.literal_eval)
df['production_companies'] = df['production_companies'].fillna("[]").apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df.head(3)
df.shape
df.to_csv("movies_tmdbMeta.csv.gz",index=False,compression="gzip")
