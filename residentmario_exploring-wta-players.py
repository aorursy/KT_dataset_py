import pandas as pd

players = pd.read_csv("../input/wta/players.csv", encoding='latin1', index_col=0)



# Top column is misaligned.

players.index.name = 'ID'

players.columns = ['First' , 'Last', 'Handidness', 'DOB', 'Country']



# Parse date data to dates.

players = players.assign(DOB=pd.to_datetime(players['DOB'], format='%Y%m%d'))



# Handidness is reported as U if unknown; set np.nan instead.

import numpy as np

players = players.assign(Handidness=players['Handidness'].replace('U', np.nan))



players.head()
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

players.Handidness.value_counts(dropna=False).plot.bar(figsize=(12, 6),

                                                       title='WTA Player Handidness')
players.set_index('DOB').resample('Y').count().Country.plot.line(

    linewidth=1, 

    figsize=(12, 4),

    title='WTA Player Year of Birth'

)
players.Country.value_counts().head(20).plot.bar(

    figsize=(12, 6),

    title='WTA Player Country Representing'

)
matches = pd.read_csv("../input/wta/matches.csv", encoding='latin1', index_col=0)

matches.head(3)
matches['winner_name'].value_counts().head(20).plot.bar(

    figsize=(12, 4),

    title='WTA Players with Most Matches Won'

)
matches['loser_name'].value_counts().head(20).plot.bar(

    figsize=(12, 4),

    title='WTA Players with Most Matches Lost'

)
pd.concat([matches['winner_name'], matches['loser_name']]).value_counts().head(20).plot.bar(

    figsize=(12, 4),

    title='WTA Players with Most Matches Played'

)
(pd.concat([matches['winner_name'], matches['loser_name']]).value_counts() < 20).astype(int).sum()
pd.Series(

    [(pd.concat([matches['winner_name'], matches['loser_name']]).value_counts() < 20)\

         .astype(int).sum(),

    (pd.concat([matches['winner_name'], matches['loser_name']]).value_counts() >= 20)\

         .astype(int).sum()],

    index=['No', 'Yes']

).plot.bar(title='Played At Least 20 Matches?')
(pd.concat([matches['winner_name'], matches['loser_name']])

     .value_counts()

     .where(lambda v: v > 20)

     .dropna()

).plot.hist(

    bins=100,

    figsize=(12, 4),

    title='WTA Career Length'

)
np.maximum.accumulate(pd.concat([matches['winner_name'], matches['loser_name']])

     .value_counts(ascending=True)

).reset_index(drop=True).plot.line()
import missingno as msno

plt.rcdefaults()

msno.matrix(matches.head(500))
plt.style.use('fivethirtyeight')



(matches

     .assign(

         winner_seed = matches.winner_seed.fillna(0).map(lambda v: v if str.isdecimal(str(v)) else np.nan),

         loser_seed = matches.loser_seed.fillna(0).map(lambda v: v if str.isdecimal(str(v)) else np.nan)

     )

     .loc[:, ['winner_seed', 'loser_seed']]

     .pipe(lambda df: df.winner_seed.astype(float) >= df.loser_seed.astype(float))

     .value_counts()

).plot.bar(title='Higher Ranked Seed Won Match')
qual = pd.read_csv("../input/wta/qualifying_matches.csv")

qual.head()
qual.shape
plt.rcdefaults()

msno.matrix(qual.head(500))
rankings = pd.read_csv("../input/wta/rankings.csv")

rankings.head()
plt.style.use('fivethirtyeight')

rankings[rankings['ranking_date'] == 20000101].ranking.sort_values().reset_index(drop=True).plot.line()
plt.style.use('fivethirtyeight')

rankings['ranking_date'].value_counts().sort_index().plot.line(linewidth=1.5)
rankings['ranking_points'].plot.hist(bins=100)
serena_williams = (rankings.query('player_id == "200033"')

     .pipe(lambda df: df.assign(ranking_date=pd.to_datetime(df.ranking_date, format='%Y%m%d', errors='coerce')))

     .set_index('ranking_date')

     .loc[:, ['ranking', 'ranking_points']]

)
fig, axarr = plt.subplots(1, 2, figsize=(12, 6))

fig.suptitle('Serena Williams Rank (L) and Points (R) Over Time')



serena_williams['ranking'].plot.line(ax=axarr[0], linewidth=2, color='steelblue')

axarr[0].set_ylim(0, 20)



serena_williams['ranking_points'].plot.line(ax=axarr[1], linewidth=2, color='steelblue')

pass