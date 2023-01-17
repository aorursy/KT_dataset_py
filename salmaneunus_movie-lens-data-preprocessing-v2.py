import pandas as pd

import numpy as np

import datetime

rating = pd.read_csv("../input/movielens-20m-dataset/rating.csv")

movie = pd.read_csv("../input/movielens-20m-dataset/movie.csv")

tag = pd.read_csv("../input/movielens-20m-dataset/tag.csv")

link = pd.read_csv("../input/movielens-20m-dataset/link.csv")

genome_tags = pd.read_csv("../input/movielens-20m-dataset/genome_tags.csv")

genome_scores = pd.read_csv("../input/movielens-20m-dataset/genome_scores.csv")
rating
movie
tag
link
genome_tags
genome_scores
rating.shape
rating.info()
rating.isnull().sum()
tag.isnull().sum()
movie.isnull().sum()
rating
rating1 = rating.copy()

rating1
rating1.info()
rating1['timestamp'].astype(str)
rating1.info()
rating1["timestamp"].dtypes
rating1['datetime'] = pd.to_datetime(rating1['timestamp'],format='%Y-%m-%d %H:%M:%S')

rating1['Hour'] = rating1['datetime'].dt.hour
rating1
rating1.info()
rating1 = rating1.drop(rating1[["datetime","timestamp"]],axis=1)

rating1
rating1.head(60)
rating1
rating2 = rating1.iloc[0:20000,0:]

rating2
rating2.to_csv('rating2.csv')