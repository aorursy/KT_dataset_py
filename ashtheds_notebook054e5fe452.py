import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import folium
chess_df=pd.read_csv('../input/top-women-chess-players/top_women_chess_players_aug_2020.csv',header=0)

chess_df.head()
chess_df.info()
chess_df.columns=chess_df.columns.astype(str).str.lower()

chess_df.head()
chess_df['inactive_flag'].value_counts(dropna=False)
chess_df['inactive_flag']=chess_df['inactive_flag'].fillna(False)

chess_df['inactive_flag']=chess_df['inactive_flag'].astype(str).replace('wi','True')

chess_df.head()
country_specific_data=chess_df.groupby('federation', as_index=False)['name'].count().sort_values('name',ascending=False)

country_specific_data.reset_index(drop=True,inplace=True)

country_specific_data.columns=['federation','number_of_players']



country_specific_data.head()
country_specific_data[0:10].plot(kind='bar',x='federation',y='number_of_players',title='Top 10 countries based on player count', grid=True)

plt.xlabel('Federation/Country')

plt.ylabel('Count of players')

plt.show()
gms_by_country=chess_df[chess_df['title']=='GM']

gms_by_country2=gms_by_country.groupby('federation')['name'].count().sort_values(ascending=False)

gms_by_country2.head(10)
gms_by_country2[0:10].plot(kind='bar',title='Top 10 countries based on the number of Grand Masters',grid=True)

plt.ylabel('count')

plt.show()
active_players=chess_df[chess_df['inactive_flag']=='False']

a=active_players.groupby('federation')['name'].count().sort_values(ascending=False)

#a.head()

a[0:5].plot(kind='bar',title='Top 10 countries based on active players',grid=True)

plt.ylabel('Count')

plt.show()
b=active_players[active_players['title']=='GM']

c=b.groupby('federation')['name'].count().sort_values(ascending=False)

c[0:10].plot(kind='bar',grid=True,title='Top 10 countries based on active GMs')

plt.ylabel('count')

plt.show()
type(c)
standard_rating=active_players.groupby('federation')['standard_rating'].mean().sort_values(ascending=False)

standard_rating[0:10].plot(kind='bar',grid=True,title='Top 10 countries based on the average Standard game rating')

plt.ylabel('Average Rating')

plt.show()
rapid_rating=active_players.groupby('federation')['rapid_rating'].mean().sort_values(ascending=False)

rapid_rating[0:10].plot(kind='bar',grid=True,title='Top 10 countries based on the average Standard game rating')

plt.ylabel('Average Rating')

plt.show()
blitz_rating=active_players.groupby('federation')['blitz_rating'].mean().sort_values(ascending=False)

blitz_rating[0:10].plot(kind='bar',grid=True,title='Top 10 countries based on the average Standard game rating')

plt.ylabel('Average Rating')

plt.show()
chess_years=active_players.groupby('year_of_birth')[['standard_rating','rapid_rating','blitz_rating']].mean()

chess_years.head()
chess_years.sort_index(ascending=True,inplace=True)