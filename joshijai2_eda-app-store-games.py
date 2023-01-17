import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
games = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv', index_col='Name')
games.head()
games.info()
games.drop(['URL', 'ID', 'Subtitle', 'Icon URL', 'In-app Purchases', 'Description', 'Developer', 'Languages', 'Primary Genre', 'Original Release Date'],axis = 1, inplace = True)
games.dropna(inplace = True)
games = games[games['User Rating Count'] > 500]
games = games[games['Price']<100]
games['Age Rating'] = games['Age Rating'].str.rstrip('+').astype('int')
games['Genres'] = games.Genres.apply(lambda x: tuple(sorted(x.split(', '))))
games['Current Version Release Date'] = games['Current Version Release Date'].apply(lambda x: x[3:]).apply(lambda x: tuple(sorted(x.split('-'))))
games['User Rating Count'] = games['User Rating Count'].apply(lambda x: x/1000000)

games['Size'] = games['Size'].apply(lambda x: x/1048576)
games.info()
games.head()
sns.jointplot('Average User Rating', 'Size', games)
sns.jointplot('Average User Rating', 'Price', games)
count = games.drop(['Price','Size','Age Rating','Average User Rating'], 1).groupby('Genres').sum()
games_genre = games.drop('User Rating Count',1).groupby('Genres').mean()
Genre_Groups = games_genre.join(count,on = 'Genres').sort_values('User Rating Count',0,False)
Genre_Groups = Genre_Groups[Genre_Groups['User Rating Count']>0.1]

Genre_Groups = Genre_Groups[Genre_Groups['Average User Rating']>4].reset_index()
Genre_Groups
plt.figure(figsize=(10,6))

sns.stripplot('Average User Rating','Genres',  data = Genre_Groups)
sns.jointplot('Price', 'Average User Rating', games)
sns.jointplot('Price', 'User Rating Count', games)