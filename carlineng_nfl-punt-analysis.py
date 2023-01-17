# kaggle/python docker image: https://github.com/kaggle/docker-python
import os
import datetime

import numpy as np
import pandas as pd
import holoviews as hv
hv.extension('bokeh')

from scipy import stats
# This dataset was created by loading data into a local Postgres server, and running the queries above.
plays = pd.read_csv("../input/punt-plays-categorized/plays_categorized.csv", sep='|')
where_kern = plays.punter_name == 'B.Kern'
not_kern = plays.punter_name != 'B.Kern'
where_hekker = plays.punter_name == 'J.Hekker'
did_punt = plays.punt_time.notnull()
is_injury = plays.injury == 1
not_injury = plays.injury == 0
plays[['injury', 'game_key']].groupby('injury').count()
plays[is_injury][['player_activity_derived', 'game_key']].groupby('player_activity_derived').count()
plays[is_injury][['player_role_type', 'game_key']].groupby('player_role_type').count()
plays[is_injury][['punt_outcome', 'game_key']].groupby('punt_outcome').count()
plays[is_injury][['season_type', 'game_key']].groupby('season_type').count()
preseason_games = 64.0
regular_season_games = 256
postseason_games = 11
print(preseason_games / (preseason_games + regular_season_games + postseason_games))
print(12/(12+25.0))
pre_punt = 12
reg_punt = 25
pre_nonpunt = 79
reg_nonpunt = 336
grand_total = pre_punt+reg_punt+pre_nonpunt+reg_nonpunt
pre_total = pre_punt+pre_nonpunt
reg_total = reg_punt+reg_nonpunt
punt_total = pre_punt+reg_punt
nonpunt_total = pre_nonpunt+reg_nonpunt

observed = [12, 79, 25, 336]
expected = [(punt_total * pre_total/float(grand_total)), 
            (nonpunt_total * pre_total/ float(grand_total)),
            (punt_total * reg_total/float(grand_total)),
            (nonpunt_total * reg_total/float(grand_total))]
stats.chisquare(f_obs=observed, f_exp=expected, ddof=2)


plays[['punter_name', 'injury']].groupby('punter_name').agg({'injury': {'num_injuries': np.sum, 'total_punts': np.size, 'injury_rate': np.mean}}).sort_values( [('injury', 'num_injuries')], ascending=False ).head(25)
punts_kern = plays[ did_punt & where_kern ]
punts_hekker = plays[ did_punt & where_hekker ]
punts_not_kern = plays[ did_punt & not_kern ]
kern_snap_punt_distance = abs(punts_kern.punter_x - punts_kern.longsnap_x)
hekker_snap_punt_distance = abs(punts_hekker.punter_x - punts_hekker.longsnap_x)
not_kern_snap_punt_distance = abs(punts_not_kern.punter_x - punts_not_kern.longsnap_x)
kern_distance_density = [hv.Distribution(kern_snap_punt_distance, label = 'Kern')]
hekker_distance_density = [hv.Distribution(hekker_snap_punt_distance, label = 'Hekker')]
not_kern_distance_density = [hv.Distribution(not_kern_snap_punt_distance, label = 'Not Kern')]
%%opts Overlay [width=600 legend_position='right']
overlay = hv.Overlay(kern_distance_density + hekker_distance_density)
overlay
print('Brett Kern mean distance from long snapper: {0}'.format(kern_snap_punt_distance.mean()))
print('Johnny Hekker mean distance from long snapper: {0}'.format(hekker_snap_punt_distance.mean()))
print('Difference: {0}'.format(hekker_snap_punt_distance.mean() - kern_snap_punt_distance.mean()))
stats.ttest_ind(kern_snap_punt_distance.dropna(), hekker_snap_punt_distance.dropna(), equal_var=False)
%%opts Overlay [width=600 legend_position='right']
overlay = hv.Overlay(kern_distance_density + not_kern_distance_density)
overlay
print('Brett Kern mean distance from long snapper: {0}'.format(kern_snap_punt_distance.mean()))
print('Rest of punters mean distance from long snapper: {0}'.format(not_kern_snap_punt_distance.mean()))
print('Difference: {0}'.format(not_kern_snap_punt_distance.mean() - kern_snap_punt_distance.mean()))
print(stats.ttest_ind(kern_snap_punt_distance.dropna(), not_kern_snap_punt_distance.dropna(), equal_var=False))
kern_snap_punt = (pd.to_datetime(plays[ did_punt & where_kern ].punt_time) - pd.to_datetime(plays[ did_punt & where_kern ].snap_time)).dt.total_seconds()
hekker_snap_punt = (pd.to_datetime(plays[ did_punt & where_hekker ].punt_time) - pd.to_datetime(plays[ did_punt & where_hekker ].snap_time)).dt.total_seconds()
not_kern_snap_punt = (pd.to_datetime(plays[ did_punt & not_kern ].punt_time) - pd.to_datetime(plays[ did_punt & not_kern ].snap_time)).dt.total_seconds()

kern_snap_punt_time = [hv.Distribution(kern_snap_punt, label = 'Kern')]
hekker_snap_punt_time = [hv.Distribution(hekker_snap_punt, label = 'Hekker')]
not_kern_snap_punt_time = [hv.Distribution(not_kern_snap_punt, label = 'Not Kern')]
print('Brett Kern time from snap to punt: {0}'.format(kern_snap_punt.mean()))
print('Johnny Hekker time from snap to punt: {0}'.format(hekker_snap_punt.mean()))
print('Difference in time from snap to punt: {0}'.format(hekker_snap_punt.mean() - kern_snap_punt.mean()))

print(stats.ttest_ind(kern_snap_punt.dropna(), hekker_snap_punt, equal_var=False))
%%opts Overlay [width=600 legend_position='right']
overlay = hv.Overlay(kern_snap_punt_time + hekker_snap_punt_time)
overlay
print('All punters but Kern: {0}'.format(not_kern_snap_punt.mean()))
print('Difference in time from snap to punt: {0}'.format(kern_snap_punt.mean() - not_kern_snap_punt.mean()))
print(stats.ttest_ind(kern_snap_punt.dropna(), not_kern_snap_punt.dropna(), equal_var=False))
%%opts Overlay [width=600 legend_position='right']
overlay = hv.Overlay(kern_snap_punt_time + not_kern_snap_punt_time)
overlay
punts_injury = plays[is_injury & did_punt]
punts_non_injury = plays[not_injury & did_punt]

# Distance
injury_snap_punt_distance = abs(punts_injury.punter_x - punts_injury.longsnap_x)
non_injury_snap_punt_distance = abs(punts_non_injury.punter_x - punts_non_injury.longsnap_x)
injury_distance_density = [hv.Distribution(injury_snap_punt_distance, label = 'injury')]
non_injury_distance_density = [hv.Distribution(non_injury_snap_punt_distance, label = 'not injury')]

# Time
injury_snap_punt_time = (pd.to_datetime(punts_injury.punt_time) - pd.to_datetime(punts_injury.snap_time)).dt.total_seconds()
non_injury_snap_punt_time = (pd.to_datetime(punts_non_injury.punt_time) - pd.to_datetime(punts_non_injury.snap_time)).dt.total_seconds()
injury_time_density = [hv.Distribution(injury_snap_punt_time, label = 'injury')]
non_injury_time_density = [hv.Distribution(non_injury_snap_punt_time, label = 'not injury')]
%%opts Overlay [width=600 legend_position='right']

overlay = hv.Overlay(injury_distance_density + non_injury_distance_density)
overlay
stats.ttest_ind(injury_snap_punt_distance.dropna(), non_injury_snap_punt_distance.dropna(), equal_var=False)
%%opts Overlay [width=600 legend_position='right']

overlay = hv.Overlay(injury_time_density + non_injury_time_density)
overlay
print('Injury punts, time from snap to punt: {0} seconds'.format(injury_snap_punt_time.mean()))
print('Non-injury punts, time from snap to punt: {0} seconds'.format(non_injury_snap_punt_time.mean()))
print('Non-injury punts were {0} seconds slower from snap to punt'.format(non_injury_snap_punt_time.mean() - injury_snap_punt_time.mean()))
print(stats.ttest_ind(injury_snap_punt_time.dropna(), non_injury_snap_punt_time.dropna(), equal_var=False))
valid_punts = plays[did_punt & plays.punter_x.notnull() & plays.longsnap_y.notnull() & plays.punt_time.notnull() & plays.snap_time.notnull()]
snap_punt_time = (pd.to_datetime(valid_punts.punt_time) - pd.to_datetime(valid_punts.snap_time)).dt.total_seconds()
snap_punt_distance =  abs(valid_punts.punter_x - valid_punts.longsnap_x)
print(stats.linregress(x=snap_punt_distance, y=snap_punt_time))
%%opts Scatter [width=600]

slope, intercept, r_value, p_value, std_err = stats.linregress(x=snap_punt_distance, y=snap_punt_time)
fitted_x = np.linspace(7, 16, 200)
fitted_y = fitted_x * slope + intercept

observed = [hv.Scatter(list(zip(snap_punt_distance, snap_punt_time)), label = 'Observed')]
fitted = [hv.Scatter(list(zip(fitted_x,fitted_y)), label = 'Fitted')]
overlay = hv.Overlay(observed + fitted)
overlay
