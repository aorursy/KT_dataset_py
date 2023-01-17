import pandas as pd



teams = pd.read_csv('/kaggle/input/nba-games/teams.csv')

games = pd.read_csv('/kaggle/input/nba-games/games.csv')

# We do not need game details, rankings, or players data.



# Format dates.

games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'], format='%Y-%m-%d')
# TODO: Cory did most of this already. Copy it over.
# TODO

# TODO: A scatterplot would fit nicely here.