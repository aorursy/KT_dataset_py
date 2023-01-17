

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json



import os

print(os.listdir("../input"))
df_reviews = pd.read_json('../input/IMDB_reviews.json', lines=True).drop_duplicates("review_text").sample(frac=1)

df_reviews.review_date = pd.to_datetime(df_reviews.review_date,infer_datetime_format=True)

df_reviews.user_id = df_reviews.user_id.astype('category').cat.codes # use int instead of string to store users

print(df_reviews.shape)

df_reviews.tail()
# easy/trivial cases. We see they're not the majority of labelled spoilers

df_reviews.review_text.str.contains("spoiler",case=False).sum()
df_reviews.is_spoiler.mean()
df = pd.read_json('../input/IMDB_movie_details.json', lines=True)

df.release_date = pd.to_datetime(df.release_date,infer_datetime_format=True)

print(df.shape)

df.tail()
 ####Very  large output file (2.9GB) if merged

# df_reviews = df_reviews.merge(df,on="movie_id",how="left",suffixes=('_review','_movie'))



# df_reviews = df_reviews.sample(frac=1)
df_reviews.to_csv("movie_spoilers_reviews.csv.gz",index=False,compression="gzip")

df.to_csv("movie_spoilers_movies.csv.gz",index=False,compression="gzip")