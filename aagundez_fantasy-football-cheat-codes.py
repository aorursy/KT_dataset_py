import numpy as np

import pandas as pd
# Passing

POINTS_PASS_TD = 4

POINTS_PASS_25_YARDS = 1

POINTS_PASS_2PT_CONVERSION = 2

POINTS_PASS_INTERCEPTED = -2



# Rushing

POINTS_RUSH_TD = 6

POINTS_RUSH_10_YARDS = 1

POINTS_RUSH_2PT_CONVERSION = 2



# Receiving

POINTS_RECEIVE_TD = 6

POINTS_RECEIVE_10_YARDS = 1

POINTS_RECEIVE_2PT_CONVERSION = 2



# Misc. Offense

POINTS_KICKOFF_RETURN_TD = 6

POINTS_PUNT_RETURN_TD = 6

POINTS_FUMBLE_RECOVERED = 6

POINTS_FUMBLE_LOST = -2
plays = pd.read_csv('../input/nflplaybyplay2015.csv')
plays['TwoPointConv'] = plays['TwoPointConv'].map({'Failure': 0, 'Success': 1})

plays['TwoPointConv'].unique()
run_plays = plays[plays['PlayType'] == 'Run']
run_plays_by_player = (run_plays[['Rusher', 'Yards.Gained', 'Touchdown', 'TwoPointConv', 'Fumble']]

    .groupby('Rusher')

    .agg({'Yards.Gained': [np.sum, np.mean], 'Touchdown': np.sum, 'TwoPointConv': np.sum, 'Fumble': np.sum}))

run_plays_by_player.fillna(0, inplace=True)
run_plays_by_player['Fumble.Points'] = run_plays_by_player['Fumble'] * POINTS_FUMBLE_LOST

run_plays_by_player['2PtConversion.Points'] = run_plays_by_player['TwoPointConv'] * POINTS_RUSH_2PT_CONVERSION

run_plays_by_player['Touchdown.Points'] = run_plays_by_player['Touchdown'] * POINTS_RUSH_TD

run_plays_by_player['Yards.Points'] = run_plays_by_player[('Yards.Gained', 'sum')] * POINTS_RUSH_10_YARDS / 10

run_plays_by_player['Total.Points'] = (run_plays_by_player['Touchdown.Points'] 

                                       + run_plays_by_player['Yards.Points']

                                       + run_plays_by_player['2PtConversion.Points']

                                       + run_plays_by_player['Fumble.Points'])

run_plays_by_player = run_plays_by_player[run_plays_by_player['Total.Points'] > 0]
run_plays_by_player.sort_values('Total.Points', ascending=False).head(n=30)
run_plays_by_player.sort_values('Total.Points', inplace=True)

run_plays_by_player['Total.Points'].plot(kind='barh', figsize=(10, 200))