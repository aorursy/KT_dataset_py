# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import necessary modules



import json

import datetime

import ast

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#importing dataset (csv file)

data = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')
data.columns
data.info()
#Converting to float and replacing '0' wit NaN

data['popularity'] = pd.to_numeric(data['popularity'], errors='coerce')

data['popularity'] = data['popularity'].replace(0, np.nan)



data['budget'] = pd.to_numeric(data['budget'], errors='coerce')

data['budget'] = data['budget'].replace(0, np.nan)



data['revenue'] = data['revenue'].replace(0, np.nan)



#Extracting Only year in release_date to make it simpler

data['year'] = pd.to_datetime(data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)



data['runtime'] = data['runtime'].astype('float')

data['vote_average'] = data['vote_average'].astype('float')

data['vote_count'] = data['vote_count'].astype('float')
data[data['budget'].notnull()][['title', 'budget', 'revenue', 'year']].sort_values('budget', ascending=False).head(10)
data[data['revenue'].notnull()][['title', 'budget', 'revenue', 'year']].sort_values('revenue', ascending=False).head(10)

year_rev = (data[data['revenue'].notnull()][['year','revenue']].groupby('year').mean())

year_rev.plot(figsize=(18,8))
data.year.value_counts().sort_index().plot(figsize=(18,8))
data[data['vote_count'].notnull()][['title','revenue', 'year','vote_count']].sort_values('vote_count', ascending=False).head(10)
data[data['vote_count'] > 3000][['title','revenue', 'year','vote_average']].sort_values('vote_average', ascending=False).head(10)
data[data['popularity'].notnull()][['title','popularity']].sort_values('popularity',ascending=False).head(10)
sns.distplot(data[(data['runtime'] < 300) & (data['runtime'] > 0)]['runtime'])
data[data['adult'] == 'True'][['title','year','vote_average']]
all_year = data.groupby('year')['title'].count()

all_year.plot(figsize=(18,5))