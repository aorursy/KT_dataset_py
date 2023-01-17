import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import re 
import time
from pandas.io.json import json_normalize
match_df = pd.read_pickle('../input/league-of-legendslol-ranked-games-2020-ver1/match_data_version1.pickle')
winner_df =  pd.read_pickle('../input/league-of-legendslol-ranked-games-2020-ver1/match_winner_data_version1.pickle')
loser_df = pd.read_pickle('../input/league-of-legendslol-ranked-games-2020-ver1/match_loser_data_version1.pickle')
match_df.info()
winner_df.info()
loser_df.info()
len(match_df['participants'].iloc[0])
'''
120 columns
teamid : 100 (team data)
championId : 7 (champion data)
'''
json_normalize(match_df['participants'].iloc[0][0])
match_df['participants'].iloc[0][0]['stats']
winner_df.columns
loser_df.columns
