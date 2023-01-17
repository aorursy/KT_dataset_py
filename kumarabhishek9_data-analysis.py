# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import itertools

import collections

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
anime_df = pd.read_csv("../input/anime.csv",header=0)

rating_df = pd.read_csv("../input/rating.csv",header = 0)
anime_df.shape
rating_df.shape
anime_df.head()
anime_df.dtypes
anime_df.isnull().sum()
#drop rows with null data

anime_df = anime_df.replace('Unknown',np.nan)

anime_df_nnull = anime_df.dropna()

anime_df_nnull.head()
#Get unique type 

anime_df_nnull.type.unique()
# anime_df_nnull['type'] is a series.

type_count_series = anime_df_nnull['type'].value_counts()

type_df = type_count_series.to_frame()

type_df
#need to reset index as to add as column

type_df = type_df.reset_index()

#rename columns from (index, type) to (type , count)

type_df.columns = ["type", "counts"]

type_df

sns.barplot(y=type_df['counts'],x=type_df['type'])
#categorize to see if members depend on type 

type_members_series = anime_df_nnull.groupby("type")['members'].agg('sum').reset_index()

type_members_df = pd.DataFrame(data = type_members_series)

type_members_df = type_members_df.sort_values("members")

sns.barplot(y=type_members_df['members'],x=type_members_df['type'])
#episode is of type object.So convert it to numeric 

anime_df_nnull['episodes'] = anime_df_nnull['episodes'].astype(int)

sns.pairplot(anime_df_nnull[['type','members','episodes','rating']], hue='type')
#type rating boxplot

sns.boxplot(data=anime_df_nnull,x='type',y='rating')
genre_values_list = anime_df_nnull["genre"].apply(lambda x : x.split(', ')).values.tolist()

genre_value_chain = itertools.chain(*genre_values_list)

#count

genre_counter = collections.Counter(genre_value_chain)

genre_df = pd.DataFrame.from_dict(genre_counter,orient='index').reset_index()

genre_df.columns = ["genre", "count"];

genre_df = genre_df.sort_values('count',ascending=False)

sns.barplot(x=genre_df['count'],y=genre_df['genre'])