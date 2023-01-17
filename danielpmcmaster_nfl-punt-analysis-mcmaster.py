# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
concussed_games = pd.read_csv("../input/video_review.csv")
play_data = pd.read_csv("../input/play_information.csv")
environmental_conditions = pd.read_csv("../input/game_data.csv")
player_data = pd.read_csv("../input/play_player_role_data.csv")
injured_players = pd.read_csv("../input/video_review.csv")

injured_players = injured_players.set_index("GSISID")
concussed_games = concussed_games.set_index("GameKey")
position_dictionary = pd.read_csv("../input/player_punt_data.csv")
position_dictionary = position_dictionary.set_index("GSISID")
player_data = player_data.set_index("GSISID")
concussion_details = concussed_games.merge(play_data, left_on=["Gamekey","PlayID"])
concussed_games.columns
concussed_games.drop(["Turnover_Related", "Friendly_Fire"], axis=1)
play_data.columns
play_data.drop(["Game_Date","Game_Clock","Play_Type","Score_Home_Visiting","PlayDescription"],axis=1)
combined = pd.merge(concussed_games,play_data, on=["GameKey","PlayID","Season_Year"] )
combined.columns
combined.drop(["Score_Home_Visiting","PlayDescription"],axis=1)
Finished = pd.merge(combined, position_dictionary, on="GSISID")
position_dictionary.columns
Finished.drop("Number",axis=1)
Finished = Finished.drop(["Home_Team_Visit_Team","Score_Home_Visiting","PlayDescription"],axis=1)
Finished.columns
NFL_Punts=Finished
import matplotlib as plt
NFL_Punts.drop_duplicates(inplace=True)
NFL_Punts[NFL_Punts.Play_Type == "Punt"].Season_Type.value_counts()
NFL_Punts[NFL_Punts.Play_Type == "Punt"].Season_Type.value_counts(normalize = True)

NFL_Punts[NFL_Punts.Play_Type == "Punt"].Position.value_counts()
print(NFL_Punts[NFL_Punts.Season_Type == "Pre"].Week)
NFL_Punts[NFL_Punts.Season_Type == "Pre"].Week.value_counts()
NFL_Punts[NFL_Punts.Season_Type == "Reg"].Week.value_counts()
NFL_Punts[NFL_Punts.Play_Type == "Punt"].Quarter.value_counts()
play_data.Quarter.value_counts(normalize=True)
NFL_Punts[NFL_Punts.Play_Type == "Punt"].Primary_Partner_Activity_Derived.value_counts()
NFL_Punts[NFL_Punts.Primary_Partner_Activity_Derived == "Blocked"].Position.value_counts()
NFL_Punts[NFL_Punts.Primary_Partner_Activity_Derived == "Blocking"].Position.value_counts()
NFL_Punts[NFL_Punts.Primary_Partner_Activity_Derived == "Tackling"].Position.value_counts()
NFL_Punts[NFL_Punts.Primary_Partner_Activity_Derived == "Tackled"].Position.value_counts()
NFL_Punts[NFL_Punts.Play_Type == "Punt"].Primary_Impact_Type.value_counts()

NFL_Punts.groupby(NFL_Punts.Primary_Impact_Type).Primary_Partner_Activity_Derived.value_counts()
