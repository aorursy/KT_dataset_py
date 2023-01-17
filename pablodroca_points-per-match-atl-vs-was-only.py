import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import glob
from subprocess import check_output

def collect_frames(files_pattern):
    frames = []
    for filepath in glob.glob(files_pattern):
        print('Collecting data from {}...'.format(filepath))
        frames.append(pd.read_csv(filepath))
    data = pd.concat(frames, ignore_index=True)
    
    #sanitize column names and print their new values.
    data.columns = data.columns.str.replace(' ', '_')
    return data


def compute_atl_was_match_score(data):
    #Only ATL vs WAS matchs will be reviewed 
    data = data.query('date == "2016-10-27" and ((home_team == "ATL" and away_team == "WAS") or (home_team == "WAS" and away_team == "ATL"))')

    #Drop data which is not necessary as MISSED shots and irrelevant columns. Sort the information for debugging
    data = data.query('current_shot_outcome == "SCORED"')
    data = data[['home_team', 'away_team', 'date', 'quarter', 'time', 'shoot_player', 'home_game', 'points']]
    data = data.sort_values(by=['date', 'quarter', 'time'])

    #Distribute points for the home or away team based in home_game
    data['home_points'] = data['home_game'].map(lambda x: 1 if x == 'Yes' else 0) * data['points']
    data['away_points'] = data['home_game'].map(lambda x: 1 if x == 'No' else 0) * data['points']

    display(data[['home_team', 'away_team', 'date']].head(1))
    display(data[['points', 'home_points', 'away_points']].sum())

data = collect_frames('../input/NBA shot log 16-17-regular season/shot log *')
atl_data = collect_frames('../input/NBA shot log 16-17-regular season/shot log ATL*')
was_data = collect_frames('../input/NBA shot log 16-17-regular season/shot log WAS*')

display(data.columns)

compute_atl_was_match_score(data)
compute_atl_was_match_score(atl_data)
compute_atl_was_match_score(was_data)
