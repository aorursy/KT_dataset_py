import numpy as np

import pandas as pd

import json

import urllib.request
def getFPLData():

    url = 'https://fantasy.premierleague.com/drf/bootstrap-static'

    

    # this works locally

    try:

        file = urllib.request.urlopen(url)

        raw_text = file.read().decode('utf-8')

        

    # this is a workaround for Kaggle kernel to run

    except:

        file = open('../input/FPL_data.json','r')

        raw_text = file.read()

        

    json_data = json.loads(raw_text)

    return json_data
FPL_json = getFPLData()
interesting_columns = ['id','name','strength_attack_home','strength_attack_away','strength_defence_home','strength_defence_away']



teams_list = []



for t in FPL_json['teams']:

    columns_list = []

    for c in interesting_columns:

        columns_list.append(t[c])

    

    teams_list.append(columns_list)



teams_df = pd.DataFrame(teams_list, columns = interesting_columns)

teams_df
interesting_columns = ['id','kickoff_time_formatted','team_h','team_a']



fixtures_list = []



for f in FPL_json['next_event_fixtures']:

    columns_list = []

    for c in interesting_columns:

        columns_list.append(f[c])

    

    fixtures_list.append(columns_list)



fixtures_df = pd.DataFrame(fixtures_list, columns = interesting_columns)

fixtures_df
temp_cols = ['team_h','name','strength_attack_home','strength_attack_away','strength_defence_home','strength_defence_away']

teams_df.columns = temp_cols

fixtures_df_join1 = fixtures_df.merge(teams_df, on='team_h')



temp_cols = ['team_a','name','strength_attack_home','strength_attack_away','strength_defence_home','strength_defence_away']

teams_df.columns = temp_cols

fixtures_df_join2 = fixtures_df_join1.merge(teams_df, on='team_a')
fixtures_df_join2['home_goals_strength_diff'] = fixtures_df_join2['strength_attack_home_x'] - fixtures_df_join2['strength_defence_away_y']

fixtures_df_join2['away_goals_strength_diff'] = fixtures_df_join2['strength_attack_away_y'] - fixtures_df_join2['strength_defence_home_x']



# Let's clear this up

interesting_columns = ['kickoff_time_formatted','name_x','home_goals_strength_diff','away_goals_strength_diff','name_y']

fixtures_final_df = fixtures_df_join2.loc[:,interesting_columns]

final_column_names = ['kickoff','home','home_score','away_score','away']

fixtures_final_df.columns = final_column_names

fixtures_final_df