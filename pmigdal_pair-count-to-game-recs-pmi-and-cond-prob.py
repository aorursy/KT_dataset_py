import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
plays_raw = pd.read_csv("../input/steam-200k.csv", header=None)

plays_raw.columns = ["player_id", "game", "action", "hours", "misc"]

plays = plays_raw.query("action == 'play'")[["player_id", "game", "hours"]]
players_per_game = plays.groupby('game').size().sort_values(ascending=False)

players_per_game = players_per_game[players_per_game >= 10]

players_per_game.head()
def coplaying(game):

    playing_some_game = plays[plays['game'] == game][["player_id"]]

    joined = playing_some_game.merge(plays, how='inner', on='player_id')

    return joined.groupby('game').size().sort_values(ascending=False)



def recommend_cond_prob(game, how_many=10):

    z = coplaying(game)

    return (z.drop(game) / z[game]).head(how_many)



def recommend_pmi(game, how_many=10):

    z = coplaying(game)

    pmi_rec = (z.drop(game) * players_per_game.sum() / z[game] / players_per_game)

    return pmi_rec.dropna().apply(np.log10).sort_values(ascending=False).head(how_many)
recommend_cond_prob("SOMA")
recommend_pmi("SOMA")
recommend_cond_prob("The Vanishing of Ethan Carter")
recommend_pmi("The Vanishing of Ethan Carter")