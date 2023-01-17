import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import matplotlib.pyplot as pyplt

import seaborn as sns



pd.options.mode.chained_assignment = None
epl = pd.read_csv('/kaggle/input/epl-results-19932018/EPL_Set.csv')

epl
epl = epl[~epl.HTHG.isnull()]
epl['HTHG'] = list(map(lambda x : int(x), epl.HTHG))

epl['HTAG'] = list(map(lambda x : int(x), epl.HTAG))
epl['Month'] = list(map(lambda x : int(x.split('/')[1]), epl.Date))
group_total = epl.groupby('HomeTeam').Div.count().reset_index()

group_total.columns = ['HomeTeam','TotalGames']

epl = pd.merge(epl,group_total,how = 'outer')
print('reversal :', np.round(np.mean(((epl.HTR == 'A') & (epl.FTR == 'H')) | ((epl.HTR == 'H') & (epl.FTR == 'A'))),2))
epl['reversal'] = 0

epl.loc[((epl.HTR == 'A') & (epl.FTR == 'H')) | ((epl.HTR == 'H') & (epl.FTR == 'A')), 'reversal'] = 1

epl
group_season = epl.groupby('Season').reversal.mean().reset_index()

plt.figure(figsize=(19,5))

plt.bar(x = group_season.Season, height = group_season.reversal)

plt.show()
group_month = epl.groupby('Month').reversal.mean().reset_index()

plt.figure(figsize=(13,5))

plt.bar(x = group_month.Month, height = group_month.reversal)

plt.show()
reversal_game = epl[epl.reversal == 1].reset_index(drop=True)



reversal_game['FinalWin'] = 0 ; reversal_game['FinalLose'] = 0

reversal_game.loc[reversal_game.FTR == 'H', 'FinalWin'] = reversal_game.HomeTeam

reversal_game.loc[reversal_game.FTR == 'A', 'FinalWin'] = reversal_game.AwayTeam

reversal_game.loc[reversal_game.FTR == 'H', 'FinalLose'] = reversal_game.AwayTeam

reversal_game.loc[reversal_game.FTR == 'A', 'FinalLose'] = reversal_game.HomeTeam



reversal_game['HDIFF'] = 0 ; reversal_game['FDIFF'] = 0

reversal_game['HDIFF'] = np.abs(reversal_game.HTHG - reversal_game.HTAG)

reversal_game['FDIFF'] = np.abs(reversal_game.FTHG - reversal_game.FTAG)

reversal_game['HowReversal'] = reversal_game.HDIFF + reversal_game.FDIFF



reversal_game
reversal_game.groupby('FTR').reversal.count()
reversal_game.pivot_table(values = 'reversal', index = 'HDIFF', columns = 'FDIFF', aggfunc = 'count', fill_value = 0 )
plt.figure(figsize=(18,5))

plt.plot(reversal_game.HowReversal, label = 'They catches by how many goal')

plt.plot(-reversal_game.HDIFF, label = 'Half-Time score difference')

plt.plot(reversal_game.FDIFF, label = 'Full-Time score difference')



plt.title('DIFFERENCE BETWEEN HALF-FULL')

plt.ylim(-5,10)

plt.legend(loc = 2)



plt.show()
group_FinalWin = reversal_game.groupby(['FinalWin']).reversal.count().reset_index()

group_FinalWin.columns = ['FinalWin','Count']

df_FinalWin = pd.merge(group_FinalWin, group_total, left_on='FinalWin', right_on = 'HomeTeam').drop(['HomeTeam'], axis = 1)

df_FinalWin['Mean'] = np.round(df_FinalWin.Count / df_FinalWin.TotalGames,3)

df_FinalWin.sort_values('Mean', ascending = False)[:20]
group_total.boxplot()

plt.show()
group_total.quantile([0.25,0.5,0.75])
q1 = float(group_total.quantile([0.25,0.5,0.75]).iloc[0])

q2 = float(group_total.quantile([0.25,0.5,0.75]).iloc[1])

q3 = float(group_total.quantile([0.25,0.5,0.75]).iloc[2])
df_FinalWin['SeperateTeam'] = 0 

df_FinalWin.loc[df_FinalWin.TotalGames < q1, 'SeperateTeam'] = 1

df_FinalWin.loc[(df_FinalWin.TotalGames >= q1) & (df_FinalWin.TotalGames < q2), 'SeperateTeam'] = 2

df_FinalWin.loc[(df_FinalWin.TotalGames >= q2) & (df_FinalWin.TotalGames < q3), 'SeperateTeam'] = 3

df_FinalWin.loc[df_FinalWin.TotalGames >= q3, 'SeperateTeam'] = 4

df_FinalWin
df1 = df_FinalWin.groupby('SeperateTeam').Count.sum().reset_index()

df2 = df_FinalWin.groupby('SeperateTeam').TotalGames.sum().reset_index()

df3 = pd.merge(df1,df2)

df3['Mean'] = df3.Count / df3.TotalGames

df3.plot.bar(x = 'SeperateTeam', y = 'Mean')

plt.show()