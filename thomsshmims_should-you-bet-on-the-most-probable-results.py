import pandas as pd

import numpy as np

import sqlite3

import matplotlib.pyplot as plt
source = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql_query("SELECT * FROM Match", source)
df.head(5)
df.columns
df['league_id'].value_counts()
# Pick essential columns from data set and add new columns needed for the research



football = df[['league_id', 'season', 'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A']].copy()

football.assign(result='', odds='', bet='')

football.head(5)
football['result'] = np.where(football['home_team_goal'] > football['away_team_goal'], 'home win',

                                        (np.where(football['home_team_goal'] < football['away_team_goal'], 'away win', 'draw')))



football['odds'] = np.where((football['B365H'] < football['B365A']) & (football['B365H'] < football['B365D']) , 'home win',

                                        (np.where(football['B365A'] < football['B365H'], 'away win', 'draw')))



football['bet'] = np.where((football['result'] == football['odds']), football.loc[:, ['B365H', 'B365D', 'B365A']].min(axis=1), -1)



football.head()
# Number of games is the same every season, great, clean data



Germany = football[football['league_id'] == 7809]

Germany['season'].value_counts()
# Splitting results by season



Germany_08_09 = Germany.loc[Germany.season == '2008/2009', :]

Germany_09_10 = Germany.loc[Germany.season == '2009/2010', :]

Germany_10_11 = Germany.loc[Germany.season == '2010/2011', :]

Germany_11_12 = Germany.loc[Germany.season == '2011/2012', :]

Germany_12_13 = Germany.loc[Germany.season == '2012/2013', :]

Germany_13_14 = Germany.loc[Germany.season == '2013/2014', :]

Germany_14_15 = Germany.loc[Germany.season == '2014/2015', :]

Germany_15_16 = Germany.loc[Germany.season == '2015/2016', :]

# Bets - the balance of gains and losses



bets = [round(Germany_08_09.bet.sum(), 2), round(Germany_09_10.bet.sum(), 2), round(Germany_10_11.bet.sum(), 2), round(Germany_11_12.bet.sum(), 2), round(Germany_12_13.bet.sum(), 2), round(Germany_13_14.bet.sum(), 2), round(Germany_14_15.bet.sum(), 2), round(Germany_15_16.bet.sum(), 2)]
balance = pd.DataFrame(bets, columns = ['balance'], index = ['08/09', '09/10', '10/11', '11/12', '12/13', '13/14', '14/15', '15/16'])



balance.head(9).transpose()
balance.plot(figsize=(20, 6))

plt.show()
ger_result = [Germany_08_09.result.value_counts().tolist(), 

                Germany_08_09.odds.value_counts().tolist(),

                Germany_09_10.result.value_counts().tolist(),

                Germany_09_10.odds.value_counts().tolist(),

                Germany_10_11.result.value_counts().tolist(),

                Germany_10_11.odds.value_counts().tolist(),

                Germany_11_12.result.value_counts().tolist(),

                Germany_11_12.odds.value_counts().tolist(),

                Germany_12_13.result.value_counts().tolist(),

                Germany_12_13.odds.value_counts().tolist(),

                Germany_13_14.result.value_counts().tolist(),

                Germany_13_14.odds.value_counts().tolist(),

                Germany_14_15.result.value_counts().tolist(),

                Germany_14_15.odds.value_counts().tolist(),

                Germany_15_16.result.value_counts().tolist(),

                Germany_15_16.odds.value_counts().tolist()]
index = ['08/09 results', '08/09 odds', '09/10 results', '09/10 odds', '10/11 results', '10/11 odds', '11/12 results', '11/12 odds', '12/13 results', '12/13 odds', '13/14 results', '13/14 odds', '14/15 results', '14/15 odds', '15/16 results', '15/16 odds']

bundesliga = pd.DataFrame(ger_result, columns = ['home win', 'away win', 'draw'], index = index)
bundesliga.transpose()
bundesliga.plot.bar(width=0.6, figsize=(20, 6))

plt.show()
#home win, draw, away win

Germany_14_15.result.value_counts()
#home win, away win, draw listing difference

Germany_14_15.odds.value_counts()
mispredictions = [(Germany_08_09.result.value_counts() - Germany_08_09.odds.value_counts()).abs().tolist(), 

                  (Germany_09_10.result.value_counts() - Germany_09_10.odds.value_counts()).abs().tolist(), 

                  (Germany_10_11.result.value_counts() - Germany_10_11.odds.value_counts()).abs().tolist(), 

                  (Germany_11_12.result.value_counts() - Germany_11_12.odds.value_counts()).abs().tolist(),

                  (Germany_12_13.result.value_counts() - Germany_12_13.odds.value_counts()).abs().tolist(),

                  (Germany_13_14.result.value_counts() - Germany_13_14.odds.value_counts()).abs().tolist(),

                  (Germany_14_15.result.value_counts() - Germany_14_15.odds.value_counts()).abs().tolist(), #glitch!

                  (Germany_15_16.result.value_counts() - Germany_15_16.odds.value_counts()).abs().tolist()]



index2 = ['08/09', '09/10', '10/11', '11/12', '12/13', '13/14', '14/15', '15/16']

mismatch = pd.DataFrame(mispredictions, columns = ['home win', 'away win', 'draw'], index = index2)
mismatch.transpose()