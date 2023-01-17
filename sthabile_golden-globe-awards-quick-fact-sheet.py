import numpy as np 

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as pl
df = pd.read_csv("../input/golden-globe-awards/golden_globe_awards.csv")

df.head()
df.tail()
print('The database contains a total of %d rows.' %df['year_film'].count())
#Narrow down the database to show the winners only

df_wins = df[df['win'] == True]
df_wins.head()
df_wins.tail()
print('This updated "wins" database contains a total of %d rows.' %df_wins['year_film'].count())
grouped = df_wins.groupby('year_award')



big_winner = grouped['nominee'].value_counts()

pd.DataFrame(big_winner[big_winner == big_winner.max()])
film_most_wins = grouped['film'].value_counts()

pd.DataFrame(film_most_wins[film_most_wins == film_most_wins.max()])
# With 5 or more wins in one ceremony

film_most_wins = grouped['film'].value_counts()

pd.DataFrame(film_most_wins[film_most_wins >= 5])
pl.figure(figsize=(6,5))

ax = pl.gca()

fs = 13

min_wins = 2

max_wins = grouped['film'].value_counts().max()

sns.set(style='darkgrid')

sns.distplot(film_most_wins[film_most_wins >= float(min_wins)], color='purple', 

             kde=0)

pl.xticks(np.arange(min_wins, max_wins+1, 1))

pl.ylabel('Frequency of Wins', fontsize=fs)

pl.xlabel('Golden Globe wins during one ceremony',fontsize=fs)

pl.show()
film_most_wins_history = df_wins.groupby('film')['win'].sum()

pd.DataFrame(film_most_wins_history[film_most_wins_history >= film_most_wins_history.max()])
film_most_wins_history = df_wins.groupby('film')['win'].sum()

min_wins_history = 3

max_wins_history = int(film_most_wins_history.max())

sns.set(style='darkgrid')

sns.distplot(film_most_wins_history[film_most_wins_history >= float(min_wins_history)], color='r', 

             kde=0)

pl.xticks(np.arange(min_wins_history, max_wins_history+1, 1))

pl.ylabel('Frequency of Wins', fontsize=fs)

pl.xlabel('Golden Globe wins in history', fontsize=fs)

pl.show()
name_most_wins = df_wins['nominee'].value_counts().idxmax()

most_wins_count = df_wins['nominee'].value_counts().max()

print('%s has received %d Golden Globe wins in total.' %(name_most_wins,most_wins_count) )
name_most_noms = df['nominee'].value_counts()

print('%s has received %d Golden Globe nominations in total.' %(name_most_noms.idxmax(),name_most_noms.max()) )
pd.DataFrame(df['nominee'].value_counts().head(10))
film_most_noms = df['film'].value_counts()

print('"%s" has received %d Golden Globe nominations in total.' %(film_most_noms.idxmax(),film_most_noms.max()) )
pd.DataFrame(df['film'].value_counts().head(10))
df[df['category'].str.contains('Cecil')]['nominee'].count()
pd.DataFrame(df[df['category'].str.contains('Cecil')]['nominee'])
df[df['category'].str.contains('Carol Burnett Award')]['nominee'].count()
pd.DataFrame(df[df['category'].str.contains('Carol Burnett Award')]['nominee'])