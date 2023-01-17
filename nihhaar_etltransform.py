import os

import numpy as np

import pandas as pd



print(os.listdir("../input"))
csv_file = "../input/books.csv"

df = pd.read_csv(csv_file, error_bad_lines=False)

df.head()
df.shape
df.dtypes
rating_df = df.groupby(['authors'])['average_rating'].mean().reset_index().round(3)

rating_df.head()
rating_df.sort_values(['average_rating', 'authors'], ascending=[False, True]).reset_index(drop=True).head(10)
pd.cut(df['# num_pages'], [-1, 100, 200, 300]).head()
range_df = df.groupby(pd.cut(df['# num_pages'], [-1, 100, 200, 300]))

range_df = range_df[['ratings_count']]

range_df.sum().reset_index()
df.groupby(['authors'])['# num_pages'].max().reset_index()
avg_df = df.groupby(['authors'])['# num_pages'].mean().reset_index()

avg_df['# num_pages'] = avg_df['# num_pages'].astype('int64')

avg_df