import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



# import module with custom plotting functions for this data

os.chdir('../input/euroleague-plotting-funcs')

import el_plotting_funcs as e

os.chdir('../../working')
# read teams advanced stats

teams_stats = pd.read_csv('../input/euroleague-basketball-advanced-stats/teams_advanced_stats.csv')



# create a columns for highlighting teams in plots

teams_stats['isMTA'] = teams_stats['team'].mask(cond=~(teams_stats['team'] == 'MTA'))



# create subsets of team_stats only for relevant seasons

teams_stats_2019 = teams_stats.loc[teams_stats['season'] == 2019]

teams_stats_2018_19 = teams_stats.loc[teams_stats['season'].isin([2018, 2019])]
# set up subplots figure and axes

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={'width_ratios': [2, 1]})



# plot net rating for teams in 2019/20

e.sorted_barplot(df=teams_stats_2019, metric='NETRtg', marked_teams=['MTA'], ax=ax0);

ax0.set_title('2019/20 ' + ax0.get_title(), fontsize=16);



# compare net rating of teams between 2018/19 and 2019/20

e.plot_parallel_pairs(df=teams_stats_2018_19, metrics=['NETRtg'], marked_iv_values=['MTA'], ax=ax1)

ax1.set_title('Net rating by season', fontsize=16);
# plot offensive vs defensive rating in 2019/20

title_suffix = '\n(Teams above the line have a positive Net Rating)'

xlabel_suffix = '\n\n<-- Worse defense\t\t\t\tBetter defense -->'.replace('\t', '    ')

ylabel_prefix = '<-- Worse offense\t\t\t\tBetter offense -->\n\n'.replace('\t', '    ')



fig, ax = e.plot_bivariate(

    df=teams_stats_2019, x='DRtg', y='ORtg', hue='isMTA', xyline=True,

    show_season=False, dont_annotate_hue_na=False, text_size='large',

    suptitle='Offensive vs Defensive Rating in 2019/20 season' + title_suffix

);

ax.set_xlabel(ax.get_xlabel() + xlabel_suffix); 

ax.set_ylabel(ylabel_prefix + ax.get_ylabel());

ax.invert_xaxis()
# set up subplots figure and axes

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={'width_ratios': [2, 1]})



# plot defensive ratings for teams in 2019/20

e.sorted_barplot(df=teams_stats_2019, metric='DRtg', marked_teams=['MTA'], ax=ax0);

ax0.set_title('2019/20 ' + ax0.get_title(), fontsize=16);



# compare teams defensive rating between 2018/19 and 2019/20

e.plot_parallel_pairs(df=teams_stats_2018_19, metrics=['DRtg'], marked_iv_values=['MTA'], ax=ax1)

ax1.set_title('Defensive rating by season', fontsize=16);

ax1.invert_yaxis()
defensive_metrics = ['OP_TS%', 'OP_AST-TOV_R', 'STLR', 'BLKR']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))



for facet_num, metric in enumerate(defensive_metrics):

    curr_ax = axes.ravel()[facet_num]

    e.sorted_barplot(

        df=teams_stats_2019, metric=metric, marked_teams=['MTA'], ax=curr_ax,

        title_size=14, tick_fontsize=10

    );

    curr_ax.set_xlabel('')

    curr_ax.set_ylabel('')