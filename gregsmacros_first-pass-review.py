# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# define method to import file and give some basic output

def filecheck(file):
    df = pd.read_csv("../input/{}".format(file))
    print(df.head())
    print(df.shape)
    return df
    
teams = filecheck("WTeams.csv")
# import the rest of the data section 1 files

seasons = filecheck("WSeasons.csv")
seeds = filecheck("WNCAATourneySeeds.csv")
# needs unique key creating to match team and their season

seeds['key'] = seeds.Season.astype(str) + seeds.TeamID.astype(str) 
print(len(seeds['key']))
print(len(seeds['key'].unique()))

seeds.head()
seasonResults = filecheck("WRegularSeasonCompactResults.csv")
# add key to match winning and losing teams

seasonResults['wKey'] = seasonResults.Season.astype(str) + seasonResults.WTeamID.astype(str) 
seasonResults['lKey'] = seasonResults.Season.astype(str) + seasonResults.LTeamID.astype(str)

seasonResults = pd.merge(seasonResults, seeds[['Seed','key']],
                         how='left', left_on='wKey', right_on='key')
seasonResults = seasonResults.rename(columns={'Seed':'wSeed'})
seasonResults = seasonResults.drop(['key'], axis = 1)

seasonResults = pd.merge(seasonResults, seeds[['Seed','key']],
                         how='left', left_on='lKey', right_on='key')
seasonResults = seasonResults.rename(columns={'Seed':'lSeed'})
seasonResults = seasonResults.drop(['key'], axis = 1)

seasonResults.head()
# pivot the table to find numbers of teams and seeds in each year
# can use this to determine how to treat unseeded teams

seasonResults.pivot_table(index='Season', values=['WTeamID','wSeed'],
                          aggfunc=lambda x: len(x.unique()))
# ((350-65)/4)/2 + 16 ~= 50 as the average "seed" for unseeded teams. This feels like it will 
# penalise too severely any losses to unseeded teams, so for now I'll use 20 in place of NaNs

seasonResults['wSeedNum'] = seasonResults.wSeed.str[1:]
seasonResults.wSeedNum = seasonResults.wSeedNum.fillna(20)
seasonResults.wSeedNum = seasonResults.wSeedNum.astype(int)

seasonResults['lSeedNum'] = seasonResults.lSeed.str[1:]
seasonResults.lSeedNum = seasonResults.lSeedNum.fillna(20)
seasonResults.lSeedNum = seasonResults.lSeedNum.astype(int)

seasonResults['seedDiff'] = seasonResults.wSeedNum - seasonResults.lSeedNum 
wins = seasonResults.pivot_table(index='wKey', values=['WScore'], aggfunc=('count','sum'))
losses = seasonResults.pivot_table(index='lKey', values=['LScore'], aggfunc=('count','sum'))

seasonSumm = pd.merge(wins, losses, left_index=True, right_index=True)

seasonSumm['gamesPlayed'] = seasonSumm['WScore']['count'] + seasonSumm['LScore']['count']
seasonSumm['record'] = seasonSumm['WScore']['count']/seasonSumm['gamesPlayed']
seasonSumm['pointsDiff'] = seasonSumm['WScore']['sum'] - seasonSumm['LScore']['sum']
seasonSumm['meanPointsDiff'] = seasonSumm['pointsDiff']/seasonSumm['gamesPlayed']

seasonSumm.columns = [' '.join(col).strip() for col in seasonSumm.columns.values]

seasonSumm.head()
tourneyResults = filecheck("WNCAATourneyCompactResults.csv")
# First add the target to the tourney result data, 1 if the Winning Team has a lower ID than the losing 
# one, and 0 otherwise

tourneyResults['target'] = (tourneyResults['WTeamID'] < tourneyResults['LTeamID']).astype(int)

# create key and merge in the features created above in the seasonSumm
tourneyResults['wKey'] = tourneyResults.Season.astype(str) + tourneyResults.WTeamID.astype(str)
tourneyResults['lKey'] = tourneyResults.Season.astype(str) + tourneyResults.LTeamID.astype(str)
tourneyResults = pd.merge(tourneyResults, seasonSumm.iloc[:, 4:],
                          how='left', left_on='wKey', right_index=True)
tourneyResults = tourneyResults.rename(columns={'gamesPlayed':'w_gamesPlayed', 
                                                'record':'w_record',
                                                'pointsDiff':'w_pointsDiff',
                                                'meanPointsDiff':'w_meanPointsDiff'})
tourneyResults = pd.merge(tourneyResults, seasonSumm.iloc[:, 4:],
                          how='left', left_on='lKey', right_index=True)
tourneyResults = tourneyResults.rename(columns={'gamesPlayed':'l_gamesPlayed', 
                                                'record':'l_record',
                                                'pointsDiff':'l_pointsDiff',
                                                'meanPointsDiff':'l_meanPointsDiff'})


print(tourneyResults.head())
print(tourneyResults.shape)
tourneyResults = pd.merge(tourneyResults, seeds[['Seed','key']],
                          how='left', left_on='wKey', right_on='key')
tourneyResults = tourneyResults.rename(columns={'Seed':'wSeed'})
tourneyResults = tourneyResults.drop(['key'], axis = 1)

tourneyResults = pd.merge(tourneyResults, seeds[['Seed','key']],
                         how='left', left_on='lKey', right_on='key')
tourneyResults = tourneyResults.rename(columns={'Seed':'lSeed'})
tourneyResults = tourneyResults.drop(['key'], axis = 1)

tourneyResults['wSeedNum'] = tourneyResults.wSeed.str[1:]
tourneyResults.wSeedNum = tourneyResults.wSeedNum.fillna(20)
tourneyResults.wSeedNum = tourneyResults.wSeedNum.astype(int)

tourneyResults['lSeedNum'] = tourneyResults.lSeed.str[1:]
tourneyResults.lSeedNum = tourneyResults.lSeedNum.fillna(20)
tourneyResults.lSeedNum = tourneyResults.lSeedNum.astype(int)

tourneyResults['seedDiff'] = tourneyResults.wSeedNum - tourneyResults.lSeedNum 
tourneyResults['playedDiff'] = tourneyResults.w_gamesPlayed - tourneyResults.l_gamesPlayed
tourneyResults['recordDiff'] = tourneyResults.w_record - tourneyResults.l_record
tourneyResults['pointsDiffDiff'] = tourneyResults.w_pointsDiff - tourneyResults.l_pointsDiff
tourneyResults['pointsRatioDiff'] = tourneyResults.w_meanPointsDiff - tourneyResults.l_meanPointsDiff
tourneyResults.to_csv('JH_augmented_tourney.csv')
seasonResults.to_csv('JH_augmented_season.csv')
seasonSumm.to_csv('JH_season_summary.csv')