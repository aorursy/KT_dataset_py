import pandas as pd



games = pd.read_csv("../input/games.csv")

games.head(1)
pd.cut(games['turns'], 20, precision=0).value_counts().sort_index().plot.bar()
games['turns'].mean(), games['turns'].median()
games['victory_status'].value_counts().plot.bar()
games['winner'].value_counts().plot.bar()
pd.concat([games['white_rating'], games['black_rating']]).plot.hist(20)
import numpy as np

pd.concat([games['white_rating'], games['black_rating']]).agg([np.mean, np.median])
games['opening_name'].value_counts()
games['opening_name'].map(lambda n: n.split("|")[0].split(":")[0]).value_counts().head(10)
games[games['opening_name'] == 'Sicilian Defense']['opening_ply'].value_counts()
games[((games['opening_name'] == 'Sicilian Defense') & (games['opening_ply'] == 5))].head(3)