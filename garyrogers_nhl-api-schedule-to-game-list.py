import pandas as pd

import json



# Used to figure out path to input data on kaggle

#import os

#os.listdir('../input')



# Pulled from https://statsapi.web.nhl.com/api/v1/schedule?season=20192020

filename = '../input/season_20192020.json'



with open(filename) as json_file:

    season_data = json.load(json_file)

dates = pd.DataFrame.from_dict(season_data['dates'])

dates.set_index('date', inplace=True)



dates.head(20)
flat_games = []



for i, d in dates.iterrows():

    for g in d['games']:

        tempObject = { "date": i, "game_id": g['gamePk'] }

        tempObject['away_team'] = g['teams']['away']['team']['name']

        tempObject['home_team'] = g['teams']['home']['team']['name']

        flat_games.append(tempObject)

        

flat_games_df = pd.DataFrame.from_dict(flat_games)

flat_games_df.set_index('game_id', inplace=True)



flat_games_df.head(20)
bruins_games = flat_games_df.loc[lambda g: (g['away_team'] == 'Boston Bruins') | (g['home_team'] == 'Boston Bruins')]



bruins_games.head(20)