import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



# import module with custom plotting functions for this data

os.chdir('../input/euroleague-plotting-funcs')

import el_plotting_funcs as e
# read teams advanced stats

teams_stats = pd.read_csv('../euroleague-basketball-advanced-stats/teams_advanced_stats.csv')



# create column for highlighting teams in offensive and defensive rating plot

fnr18_cond = (teams_stats['team'] == 'FNR') & (teams_stats['season'] == 2018)

mta19_cond = (teams_stats['team'] == 'MTA') & (teams_stats['season'] == 2019)

pts_rtg_disply_cond = fnr18_cond | mta19_cond

teams_stats.loc[pts_rtg_disply_cond, 'pts_rtg_disply'] = teams_stats.loc[pts_rtg_disply_cond, 'team']



# create columns for highlighting teams in win% plot

win_display_cond = (teams_stats['team'].isin(['ASV', 'ZAL'])) & (teams_stats['season'] == 2019)

teams_stats.loc[win_display_cond, 'win%_disply'] = teams_stats.loc[win_display_cond, 'team']
# set up figure and axes

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [2, 1]})



# plot net rating vs win%

e.plot_bivariate(

    df=teams_stats, x='NETRtg', y='win_pct', hue='win%_disply', fit_reg=True, ax=ax0

);

ax0.set_title('Predicting win% from Net Rating\nin seasons 2016/17-2019/20', fontsize=16);



# compare teams position in eurloeague by win% vs net rating

e.plot_parallel_pairs(

    df=teams_stats.loc[teams_stats['season'] == 2019], kind='ranking',

    metrics=['win_pct', 'NETRtg'], iv='team', annotate_only_marked=True,

    marked_iv_values=['ASV', 'ZAL'], ax=ax1

);

ax1.set_title('2019/20 league position\nby win% vs. net rating', fontsize=16);
fig, ax = e.plot_bivariate(

    df=teams_stats, x='PTS40', y='ORtg', hue='pts_rtg_disply', fit_reg=True, text_size='large', suptitle_size=16,

    suptitle='Points scored per 40 minutes vs offensive rating\n(Highlight Maccabi in 2019/20 Vs Fener in 2018/19)'

);

ax.set_ylabel('Offensive Rating\n(Points scored per 100 possessions)');
fig, ax = e.plot_bivariate(

    df=teams_stats, x='OP_PTS40', y='DRtg', hue='pts_rtg_disply', fit_reg=True, text_size='large', suptitle_size=16,

    suptitle='Points conceded per 40 minutes vs defensive rating\n(Highlight Maccabi in 2019/20 Vs Fener in 2018/19)'

);

ax.set_ylabel('Defensive Rating\n(Points conceded per 100 possessions)');

ax.invert_yaxis()
# plot pace of all euroleague teams in 2018/19 and 2019/20

e.sorted_barplot(

    df=teams_stats.loc[teams_stats['season'].isin([2018, 2019])].copy(), metric='PACE',

    marked_teams=['MTA', 'FNR'], show_season=True, figsize=(15, 5), tick_fontsize=10, tick_rot=60

);

plt.ylim(55, 80);

plt.title('PACE (possessions per 40 minutes) in seasons 2018/19-2019/20', fontsize=16);