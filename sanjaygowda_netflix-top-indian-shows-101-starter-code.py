#####Import librarys 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
netflixs = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

netflix_datas = netflixs.copy()
netflix_datas.head()
#### Shape of data

netflix_datas.shape
#### sum of null values

netflix_datas.isnull().sum()
(netflix_datas.isnull().sum()/netflix_datas.shape[0])*100
netflix2 =netflix_datas.copy()



netflix2 = netflix2.fillna(netflix2.mode().iloc[0])
netflix2.isnull().sum()
#### all stats operation - mean , SD,range,quartile



netflix2.describe(include='all')
#### data types 

netflix2.dtypes
####total info of data

netflix2.info()
netflix2.show_id.nunique()
netflix2.listed_in.nunique()
netflix2['listed_in'].value_counts()
netflix2['listed_in'].value_counts().describe()
genres =netflix2['listed_in'].value_counts()
outliers = []

def find_outliers(data):

    

    max_std=3

    mean  = np.mean(data)

    std  = np.std(data)

    

    

    for i in data:

        z_score=(i - mean)/std

        if np.abs(z_score)> max_std:

            outliers.append(i)

    return outliers        
outlier_are = find_outliers(genres)
outlier_are
#### Outlies are the popular ganres

sns.countplot(netflix2['listed_in'])
movies = netflix2[netflix2['type'] == 'Movie']

tv_shows =netflix2[netflix2['type'] == 'TV Show']
### total num of movies

movies.show_id.count().sum()
### total num of tv_shows

tv_shows.show_id.count().sum()
(netflix2['type'].value_counts()/netflix2.shape[0])*100
sns.countplot(netflix2['type'])

plt.title('Show Type')
shows_by_years=netflix2.release_year.value_counts()[:15]



shows_by_years
shows_by_years.plot('bar')

plt.title('Shows add years ')
plt.figure(1, figsize=(15, 7))

plt.title("Country with maximum shows")

sns.countplot(x = "country",hue='type', order=netflix2['country'].value_counts().index[0:15] ,data=netflix2,palette='Accent')
india=netflix2[netflix2["country"]=="India"]

india.show_id.count().sum()
india['type'].unique()
sns.countplot(data=india,x="type")
india['rating'].value_counts()
sns.countplot(data=india,x="rating")
india['listed_in'].value_counts()[0:10]
india['director'].value_counts()[0:10]
india['cast'].value_counts()[0:10]