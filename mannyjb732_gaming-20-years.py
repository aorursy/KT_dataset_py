#import neccessary packages 



import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

from pandas import DataFrame

import seaborn as sns 

from datetime import datetime



%matplotlib inline 

#import data 



games = pd.read_csv("../input/ign.csv")



games.head()
games.isnull().sum()
games.drop(['Unnamed: 0', 'url'], axis=1, inplace=True)

games.drop(games.index[516],inplace=True)
games.describe()
games['date'] = pd.to_datetime(games.release_year*10000+games.release_month*100+games.release_day,format='%Y%m%d')



games.head()

plt.figure(figsize=(10,8))

games['genre'].value_counts()[:10].plot(kind='pie',autopct='%1.1f%%',shadow=True, explode=[0.1,0,0,0,0,0,0,0,0,0])

plt.show()



plt.figure(figsize=(10,8))

games['platform'].value_counts()[:20].plot(kind='bar')

plt.show()
plt.figure(figsize=(10,8))

sns.boxplot('release_year', 'score', data = games)

plt.show()
platform = [ 'Wii U', 'PlayStation 4', 'Xbox One', 'Xbox 360']



data = games[games['platform'].isin(platform)]





t = data.groupby(['date', 'platform'], as_index=False).mean()

        

plt.figure(figsize=(23,15))

a = sns.FacetGrid(t, col='platform', col_wrap=2, size=3, aspect=2.9, hue='platform')

a = a.map(plt.plot, 'date', 'score')

plt.show()


