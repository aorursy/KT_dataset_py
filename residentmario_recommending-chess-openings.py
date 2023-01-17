import pandas as pd

games = pd.read_csv("../input/games.csv")

games.head(3)
games = (

    games.assign(

        opening_archetype=games.opening_name.map(

            lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()

        ),

        opening_moves=games.apply(lambda srs: srs['moves'].split(" ")[:srs['opening_ply']],

                                  axis=1)

    )

)
games_played = pd.concat([games['white_id'], games['black_id']]).value_counts()

games_played.reset_index(drop=True).plot.line(figsize=(16, 8), fontsize=18)



n_ge_2 = len(games_played[games_played > 1])

print(str(n_ge_2) + " players who have played at least two games.")



import matplotlib.pyplot as plt

plt.axvline(n_ge_2, color='green')



len(games_played)
games_played[games_played > 1].sum()
opening_used = (pd.concat([

                   games.groupby('white_id')['opening_archetype'].value_counts(),

                   games.groupby('black_id')['opening_archetype'].value_counts()

                ])

                    .rename(index='openings_used')

                    .reset_index()

                    .rename(columns={'white_id': 'player_id', 'openings_used': 'times_used'})

                    .groupby(['player_id', 'opening_archetype']).sum()

               )
opening_used.head(10)
(opening_used

     .reset_index()

     .groupby('player_id')

     .filter(lambda df: df.opening_archetype.isin(['Sicilian Defense']).any())

     .query('opening_archetype != "Sicilian Defense"')

     .groupby('opening_archetype')

     .times_used

     .sum()

     .sort_values(ascending=False)

     .to_frame()

     .pipe(lambda df: df.assign(times_used = df.times_used / df.times_used.sum()))

     .squeeze()

     .head(10)

)
(opening_used

     .reset_index()

     .groupby('player_id')

     .filter(lambda df: df.opening_archetype.isin(["King's Gambit"]).any())

     .query('opening_archetype != "King\'s Gambit"')

     .groupby('opening_archetype')

     .times_used

     .sum()

     .sort_values(ascending=False)

     .to_frame()

     .pipe(lambda df: df.assign(times_used = df.times_used / df.times_used.sum()))

     .squeeze()

     .head(10)

)
import numpy as np



def threshold_map(n_opening, n_all):

    if pd.isnull(n_opening):

        return np.nan

    elif n_opening / n_all >= 1 / 4:

        return 5

    elif n_opening / n_all >= 1 / 8:

        return 4

    elif n_opening / n_all > 1 / 16:

        return 3

    else:

        return 2



recommendations = opening_used.unstack(-1).loc[:, 'times_used'].apply(

    lambda srs: srs.map(lambda v: threshold_map(v, srs.sum())),

    axis='columns'

)
recommendations.head()
pd.Series(recommendations.values.flatten()).value_counts().sort_index().plot.bar()
from sklearn.metrics.pairwise import pairwise_distances

# user_similarity = pairwise_distances(train.fillna(0), metric='cosine')

item_similarity = pairwise_distances(recommendations.T.fillna(0), metric='cosine')
item_similarity.shape
correction = np.array([np.abs(item_similarity).sum(axis=1)])

item_predictions = recommendations.fillna(0).dot(item_similarity).apply(

    lambda srs: srs / np.array([item_similarity.sum(axis=1)]).flatten(), axis='columns')
item_predictions.head()
recommended_opening_numbers = item_predictions.apply(

    lambda srs: np.argmax(srs.values), axis='columns'

)

recommended_opening_numbers.head()
opening_names = pd.Series(recommendations.columns)

recommended_openings = recommended_opening_numbers.map(opening_names)
recommended_openings.head()
recommended_openings.value_counts().head(20).iloc[1:].plot.bar(figsize=(24, 10), fontsize=22)