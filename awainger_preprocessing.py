import feather
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import tqdm

%matplotlib inline
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
play_information = pd.read_csv('../input/play_information.csv') # One row per punt

# Play Description Features
PlayDescription_split = play_information.PlayDescription.str.split("punts")
play_information['PlayDescription_last'] = PlayDescription_split.apply(lambda x: x[-1])
play_information['Has_Punt'] = PlayDescription_split.apply(lambda x: len(x) > 1)
play_information['Has_Fair_Catch'] = play_information.PlayDescription_last.str.contains('fair catch')
play_information['Punt_Distance'] = play_information.PlayDescription_last.str.extract('^ ([0-9]+) yard').astype('float')
play_information['Has_Muff'] = play_information.PlayDescription_last.str.contains('MUFFS')
play_information['Has_Penalty'] = play_information.PlayDescription_last.str.contains('PENALTY')
play_information['Has_Return'] = (
    play_information.Has_Punt & (
        play_information.PlayDescription_last.str.contains('for -?(?:[0-9]+ yard|no gain)', regex=True)
        | play_information.Has_Muff
    )
)

play_information['Punt_Type'] = play_information.apply(
    lambda row:
        np.NaN if not row.Has_Punt
        else (
            'fair catch' if row.Has_Fair_Catch
            else (
                'return' if row.Has_Return
                else 'unreturnable'
            )
        )
    , axis=1
)

def extract_punt_return_length(row):
    if row.Punt_Type == 'unreturnable':
        return np.nan
    elif row.Punt_Type == 'fair catch':
        return 0.0
    elif 'for no gain' in row.PlayDescription_last:
        return 0.0
    else:
        try:
            return float(re.search('for (-?[0-9]+) yard', row.PlayDescription_last).group(1))
        except:
            return 0.0

play_information['Punt_Return_Length'] = play_information.apply(
    lambda row: extract_punt_return_length(row), axis=1
)

# Time Features
play_information['Game_Clock_Min'] = play_information.Game_Clock.str.extract('([0-9]+):[0-9]+').astype('int16')
play_information['Game_Clock_Sec'] = play_information.Game_Clock.str.extract('[0-9]+:([0-9]+)').astype('int16')
play_information['Time_Passed_Sec'] = play_information.apply(
    lambda row: 
        (900 * (row['Quarter'] - 1)) +
        (60 * (15 - (row['Game_Clock_Min'] + 1))) +
        (60 - row['Game_Clock_Sec'])
    , axis=1
)

# Score Features
play_information['Home_Team'] = play_information.Home_Team_Visit_Team.str.extract('([A-Z]+)-[A-Z]+')
play_information['Away_Team'] = play_information.Home_Team_Visit_Team.str.extract('[A-Z]+-([A-Z]+)')
play_information['Home_Score'] = play_information.Score_Home_Visiting.str.extract('([0-9]+) - [0-9]+').astype('int16')
play_information['Away_Score'] = play_information.Score_Home_Visiting.str.extract('[0-9]+ - ([0-9]+)').astype('int16')
play_information['Score_Differential'] = play_information.apply(
    lambda row: 
        row.Home_Score - row.Away_Score 
        if row.Poss_Team == row.Home_Team 
        else row.Away_Score - row.Home_Score
    , axis=1
)

# Yard Line Features
play_information['Yard_Line_Team'] = play_information.YardLine.str.extract('([A-Z]+) [0-9]+')
play_information['Yard_Line_Num'] = play_information.YardLine.str.extract('[A-Z]+ ([0-9]+)').astype('int16')
play_information['Yard_Line_Absolute'] = play_information.apply(
    lambda row:
        row.Yard_Line_Num 
        if row.Yard_Line_Team == row.Poss_Team
        else 100 - row.Yard_Line_Num
    , axis=1
)

play_information.to_feather('play_information.feather')
game_data = pd.read_csv('../input/game_data.csv') # One row per game
game_data.loc[game_data.Stadium == 'Hard Rock Stadium', 'Turf'] = 'Natural Grass'
game_data['Is_Grass'] = game_data.Turf.str.strip().str.lower().str.contains('grass|natural')
game_data['StadiumType'] = game_data.StadiumType.fillna('Outdoor')
game_data['Is_Outdoor'] = game_data.StadiumType.str.lower().str.strip().str.contains('out|open|heinz|oudoor|ourdoor')
game_data.loc[~game_data.Is_Outdoor, 'Temperature'] = 70.0
game_data.to_feather('game_data.feather')
play_player_role_data = pd.read_csv('../input/play_player_role_data.csv')

# Mapping punt positions to Kicking and Receiving team
role_metadata = {
    'PR': {'Role_Team': 'R', 'Super_Role': 'Returner'},
    'PDL1': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'PDR1': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'PRG': {'Role_Team': 'K', 'Super_Role': 'Coverage Lineman'},
    'P': {'Role_Team': 'K', 'Super_Role': 'Punter'},
    'PLG': {'Role_Team': 'K', 'Super_Role': 'Coverage Lineman'},
    'PRT': {'Role_Team': 'K', 'Super_Role': 'Coverage Lineman'},
    'PLS': {'Role_Team': 'K', 'Super_Role': 'Coverage Lineman'},
    'PLT': {'Role_Team': 'K', 'Super_Role': 'Coverage Lineman'},
    'PLW': {'Role_Team': 'K', 'Super_Role': 'Coverage Wing'},
    'PDR2': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'PRW': {'Role_Team': 'K', 'Super_Role': 'Coverage Lineman'},
    'PDL2': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'GL': {'Role_Team': 'K', 'Super_Role': 'Gunner'},
    'GR': {'Role_Team': 'K', 'Super_Role': 'Gunner'},
    'PDL3': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'PDR3': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'VL': {'Role_Team': 'R', 'Super_Role': 'Return Corner'},
    'VR': {'Role_Team': 'R', 'Super_Role': 'Return Corner'},
    'PPR': {'Role_Team': 'K', 'Super_Role': 'Coverage Protector'},
    'PLL': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'},
    'PPL': {'Role_Team': 'K', 'Super_Role': 'Coverage Protector'},
    'PLR': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'},
    'VRo': {'Role_Team': 'R', 'Super_Role': 'Return Corner'},
    'VRi': {'Role_Team': 'R', 'Super_Role': 'Return Corner'},
    'VLi': {'Role_Team': 'R', 'Super_Role': 'Return Corner'},
    'VLo': {'Role_Team': 'R', 'Super_Role': 'Return Corner'},
    'PDL4': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'PDR4': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'PLM': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'},
    'PLR1': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'},
    'PLR2': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'},
    'PLL2': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'},
    'PLL1': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'},
    'PFB': {'Role_Team': 'R', 'Super_Role': 'Return Protector'},
    'PDL5': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'PDR5': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'GRo': {'Role_Team': 'K', 'Super_Role': 'Gunner'},
    'GRi': {'Role_Team': 'K', 'Super_Role': 'Gunner'},
    'PDM': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'GLi': {'Role_Team': 'K', 'Super_Role': 'Gunner'},
    'GLo': {'Role_Team': 'K', 'Super_Role': 'Gunner'},
    'PDL6': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'PLR3': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'},
    'PLL3': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'},
    'PPLi': {'Role_Team': 'K', 'Super_Role': 'Coverage Protector'},
    'PPLo': {'Role_Team': 'K', 'Super_Role': 'Coverage Protector'},
    'PC': {'Role_Team': 'K', 'Super_Role': 'Coverage Protector'},
    'PDR6': {'Role_Team': 'R', 'Super_Role': 'Return Lineman'},
    'PPRi': {'Role_Team': 'K', 'Super_Role': 'Coverage Protector'},
    'PPRo': {'Role_Team': 'K', 'Super_Role': 'Coverage Protector'},
    'PLM1': {'Role_Team': 'R', 'Super_Role': 'Return Linebacker'}
}

play_player_role_data['Role_Team'] = play_player_role_data.Role.apply(lambda role: role_metadata[role]['Role_Team'])
play_player_role_data['Super_Role'] = play_player_role_data.Role.apply(lambda role: role_metadata[role]['Super_Role'])
play_player_role_data.to_feather('play_player_role_data.feather')
video_review = pd.read_csv('../input/video_review.csv')
video_review = video_review.merge(
    play_player_role_data,
    on=['GameKey', 'PlayID', 'GSISID'], how='left',
    validate='one_to_one'
)

video_review[
    ['GameKey', 'PlayID', 'GSISID', 'Role', 'Role_Team', 'Super_Role']
].to_feather('video_review.feather')
dtypes = {'Season_Year': 'int16',
         'GameKey': 'int16',
         'PlayID': 'int16',
         'GSISID': 'float32',
         'Time': 'str',
         'x': 'float32',
         'y': 'float32',
         'dis': 'float32',
         'o': 'float32',
         'dir': 'float32',
         'Event': 'str'}

col_names = list(dtypes.keys())

path = '../input/'
ngs_files = ['NGS-2016-pre.csv',
             'NGS-2016-reg-wk1-6.csv',
             'NGS-2016-reg-wk7-12.csv',
             'NGS-2016-reg-wk13-17.csv',
             'NGS-2016-post.csv',
             'NGS-2017-pre.csv',
             'NGS-2017-reg-wk1-6.csv',
             'NGS-2017-reg-wk7-12.csv',
             'NGS-2017-reg-wk13-17.csv',
             'NGS-2017-post.csv']

df_list = []

for f in tqdm.tqdm(ngs_files):
    df = pd.read_csv(path + f, usecols=col_names,dtype=dtypes)
    
    df_list.append(df)
# Merge all dataframes into one dataframe
ngs = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list
gc.collect()

# Convert Time to datetime
ngs['Time'] = pd.to_datetime(ngs['Time'], format='%Y-%m-%d %H:%M:%S')

# Convert season year to 0/1
ngs['Season_Year'] = ngs['Season_Year'].astype('category').cat.codes

# Fill NA values then downcast to int32
ngs['GSISID'] = ngs['GSISID'].fillna(-1).astype('int32')

# Convert o and dir to int16 (don't need that level of precision)
ngs['o'] = ngs['o'].astype('int16')
ngs['dir'] = ngs['dir'].astype('int16')

# Write to feather
ngs.reset_index(drop=True).to_feather('ngs.feather')
catch_events = ngs[ngs.Event.isin(['fair_catch', 'punt_received'])].merge(
    play_player_role_data, on=['GameKey', 'PlayID', 'GSISID'], how='left', validate='many_to_one'
)

punt_returner = catch_events[catch_events.Role == 'PR'][
    ['GameKey', 'PlayID', 'GSISID', 'x', 'y', 'Event', 'Role', 'Super_Role']
]
kicking_team = catch_events[catch_events.Role_Team == 'K'][
    ['GameKey', 'PlayID', 'GSISID', 'x', 'y', 'Event', 'Role', 'Super_Role']
]

pr_cross_kick = punt_returner.merge(
    kicking_team, on=['GameKey', 'PlayID', 'Event'], how='left', validate='many_to_many',
    suffixes=['_pr', '_k']
)

pr_cross_kick['Coverage_Distance'] = (
    (
        (pr_cross_kick['x_pr'] - pr_cross_kick['x_k']) ** 2
    ) + (
        (pr_cross_kick['y_pr'] - pr_cross_kick['y_k']) ** 2
    )
) ** .5

min_distances = pr_cross_kick.loc[pr_cross_kick.groupby(['GameKey', 'PlayID'])['Coverage_Distance'].idxmin()]
min_distances.reset_index(drop=True).to_feather('min_distances.feather')

pr_cross_kick_2 = pr_cross_kick.drop(pr_cross_kick.groupby(['GameKey', 'PlayID'])['Coverage_Distance'].idxmin())
second_min_distances = pr_cross_kick_2.loc[
    pr_cross_kick_2.groupby(['GameKey', 'PlayID'])['Coverage_Distance'].idxmin()
]

second_min_distances.reset_index(drop=True).to_feather('second_min_distances.feather')
punt_time = ngs[ngs.Event.isin(['punt'])].groupby(['GameKey', 'PlayID'])['Time'].min().reset_index()
receive_time = ngs[ngs.Event.isin(['fair_catch', 'punt_received'])].groupby(['GameKey', 'PlayID'])['Time'].min().reset_index()

punt_to_reception = punt_time.merge(
    receive_time, on=['GameKey', 'PlayID'], how='inner', validate='one_to_one', suffixes=['_punt', '_receive']
)
punt_to_reception['Hangtime'] = (punt_to_reception['Time_receive'] - punt_to_reception['Time_punt']).dt.total_seconds()

punt_to_reception.to_feather('punt_hangtime.feather')

ngs_punt_to_reception = ngs.merge(
    punt_to_reception, on=['GameKey', 'PlayID'], how='inner', validate='many_to_one'
)[['GameKey', 'PlayID', 'GSISID', 'Time', 'x', 'y', 'dis', 'Time_punt', 'Time_receive', 'Hangtime']]

ngs_punt_to_reception_filtered = ngs_punt_to_reception[
    (ngs_punt_to_reception.Time >= ngs_punt_to_reception.Time_punt) &
    (ngs_punt_to_reception.Time <= ngs_punt_to_reception.Time_receive)
]

punt_to_reception_speed = ngs_punt_to_reception_filtered.groupby(
    ["GameKey", 'PlayID', 'GSISID']
).agg({"dis": np.sum, "Hangtime": np.min}).reset_index()
punt_to_reception_speed['Yards_Per_Second'] = punt_to_reception_speed.dis / punt_to_reception_speed.Hangtime
snap_time = ngs[ngs.Event == 'ball_snap'].groupby(['GameKey', 'PlayID'])['Time'].min().reset_index()
snap_to_punt = snap_time.merge(
    punt_time, on=['GameKey', 'PlayID'], how='inner', validate='one_to_one', suffixes=['_snap', '_punt']
)

snap_to_punt['Snap_To_Punt_time'] = (snap_to_punt['Time_punt'] - snap_to_punt['Time_snap']).dt.total_seconds()
snap_to_punt = snap_to_punt[(snap_to_punt.Snap_To_Punt_time > 0) & (snap_to_punt.Snap_To_Punt_time < 5)]

punt_to_reception_speed = punt_to_reception_speed.merge(
    snap_to_punt[['GameKey', 'PlayID', 'Snap_To_Punt_time']],
    on=['GameKey', 'PlayID'], how='inner', validate='many_to_one'
)
adjusted_coverage_distances = pr_cross_kick.merge(
    punt_to_reception_speed,
    left_on=['GameKey', 'PlayID', 'GSISID_k'], right_on=['GameKey', 'PlayID', 'GSISID'],
    how='inner', validate='many_to_one'
)

adjusted_coverage_distances.to_feather('adjusted_coverage_distances.feather')
