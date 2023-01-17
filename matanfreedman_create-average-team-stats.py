# Load libraries:

import pandas as pd

import numpy as np

import os





pd.options.display.max_rows=99

pd.options.display.max_columns=999
# load data, first I loaded in only a few rows to get the infered dtypes from pandas

season_results = f'../input/datafiles/RegularSeasonDetailedResults.csv'

season_df = pd.read_csv(season_results, nrows=100)



season_dtypes = season_df.dtypes.to_dict()

season_df = pd.read_csv(season_results, dtype = season_dtypes, low_memory=False)

season_df.head()
# collect stats for each team:

#stat columns:

w_cols = ['Season', 'WTeamID', 'WFGM', 'WFGA','WFGM3','WFGA3','WFTM','WFTA',

          'WOR','WDR','WAst','WTO','WStl','WBlk','WPF']



l_cols = ['Season', 'LTeamID', 'LFGM','LFGA','LFGM3','LFGA3','LFTM','LFTA',

          'LOR','LDR','LAst','LTO','LStl','LBlk','LPF']



#collecting stats:

# average stats per team for winning game

WTeam_ID_mean = season_df[w_cols].groupby(['Season', 'WTeamID']).agg({'mean'}).reset_index()



# number wins per team

WTeam_counts = season_df.groupby(['Season', 'WTeamID']).agg({'WScore':'count'}).reset_index()

# average stats per team for lossing games

LTeam_ID_mean = season_df[l_cols].groupby(['Season', 'LTeamID']).agg({'mean'}).reset_index()

 # number losses per team

LTeam_counts = season_df.groupby(['Season', 'LTeamID']).agg({'LScore':'count'}).reset_index()



# rename columns:

col_names = ['Season', 'TeamID', 'FGM', 'FGA','FGM3','FGA3','FTM','FTA','OR',

             'DR','Ast','TO','Stl','Blk','PF']

WTeam_ID_mean.columns = col_names

LTeam_ID_mean.columns = col_names



# rename columns for win/loss counts

WTeam_counts.columns = ['Season', 'TeamID', 'Wins']

LTeam_counts.columns = ['Season', 'TeamID', 'Losses']



# merge number of wins and losses with their respective average stats

 

#indices should be the same

WTeam_ID_mean = WTeam_ID_mean.merge(WTeam_counts, how='left', on=None)

LTeam_ID_mean = LTeam_ID_mean.merge(LTeam_counts, how='left', on=None)



# weighted mean:

cols = ['FGM', 'FGA','FGM3','FGA3','FTM','FTA','OR', 'DR','Ast','TO','Stl','Blk','PF']

stats_wins = WTeam_ID_mean[cols].mul(WTeam_ID_mean['Wins'], axis=0) # multiply by number of wins

stats_losses = LTeam_ID_mean[cols].mul(LTeam_ID_mean['Losses'], axis=0)



# sum of total games:

games = WTeam_ID_mean['Wins'] + LTeam_ID_mean['Losses']



# weighted mean calculation:

team_stats = (stats_wins + stats_losses).div(games, axis=0)



# merge stats with year and ID, should still have same index

final_stats = LTeam_ID_mean[['Season', 'TeamID']].merge(team_stats, 

                                                        how='left', 

                                                        right_index=True, 

                                                        left_index=True) 

final_stats.head()