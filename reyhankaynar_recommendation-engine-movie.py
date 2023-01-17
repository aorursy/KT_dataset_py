# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import csv

import surprise

from surprise import Reader, Dataset

from surprise import accuracy

from surprise import SVD

import random

import matplotlib.pyplot as plt # data visualization library

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS #used to generate world cloud

import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
df1= pd.read_csv('../input/movies.csv')

df1.shape
df1.head(10)
df2= pd.read_csv('../input/ratings.csv')

df2.shape
df2.head(10)
df3= pd.read_csv('../input/links.csv')

df3.shape

df3.head()
title= pd.read_csv('../input/movies.csv')

title.head(10)
df2 = pd.merge(df2,title, on='movieId')

df2.head(20)
df1.describe()
df2.info()
df2.describe()
ratings = pd.DataFrame(df2.groupby('title')['rating'].mean())

ratings.head(10)
ratings['number_of_ratings'] = df2.groupby('title')['rating'].count()

ratings.head(10)
import matplotlib.pyplot as plt

%matplotlib inline

ratings['rating'].hist(bins=50)
ratings['number_of_ratings'].hist(bins=60)
import seaborn as sns

sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
movie_matrix = df2.pivot_table(index='userId', columns='title', values='rating')

movie_matrix.head(50)
ratings.sort_values('number_of_ratings', ascending=False).head(10)
AFO_user_rating = movie_matrix['Air Force One (1997)']

contact_user_rating = movie_matrix['Contact (1997)']
AFO_user_rating.head()

contact_user_rating.head(20)
similar_to_air_force_one=movie_matrix.corrwith(AFO_user_rating)
similar_to_air_force_one.head(20)
similar_to_contact = movie_matrix.corrwith(contact_user_rating)
similar_to_contact.head(20)
corr_contact = pd.DataFrame(similar_to_contact, columns=['Correlation'])

corr_contact.dropna(inplace=True)

corr_contact.head()

corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['correlation'])

corr_AFO.dropna(inplace=True)

corr_AFO.head()
corr_AFO = corr_AFO.join(ratings['number_of_ratings'])

corr_contact = corr_contact.join(ratings['number_of_ratings'])

corr_AFO .head()

corr_contact.head()
corr_AFO[corr_AFO['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10)
corr_contact[corr_contact['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10)
from surprise import Reader, Dataset, SVD, evaluate

reader = Reader()

ratings = pd.read_csv('../input/ratings.csv')

ratings.head(10)
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

data.split(n_folds=5)
svd = SVD()

evaluate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()

svd.fit(trainset)
ratings[ratings['userId'] == 1]
svd.predict(1, 302, 3)