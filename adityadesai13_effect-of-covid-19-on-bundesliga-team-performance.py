import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
game_data = pd.read_csv(r'/kaggle/input/european-football-database-20192020/D1.csv')
game_data
# removing extraneous data - just want to start with win rate analysis. Will do other info later

win_data = game_data[['Date','HomeTeam','AwayTeam', 'FTHG', 'FTAG','FTR']]
win_data
# Adjusting the info to make the upcoming steps simpler

pd.set_option('mode.chained_assignment', None)
win_data['HomeWin'] = win_data['FTR'].apply(lambda x: 1 if x is 'H' else 0)
win_data['HomeLoss'] = win_data['FTR'].apply(lambda x: 1 if x is 'A' else 0)
win_data['HomeDraw'] = win_data['FTR'].apply(lambda x: 1 if x is 'D' else 0)

win_data['AwayWin'] = win_data['FTR'].apply(lambda x: 1 if x is 'A' else 0)
win_data['AwayLoss'] = win_data['FTR'].apply(lambda x: 1 if x is 'H' else 0)
win_data['AwayDraw'] = win_data['FTR'].apply(lambda x: 1 if x is 'D' else 0)
# win_data = win_data[['Date','HomeTeam', 'AwayTeam', 'HomeWin', 'HomeLoss','HomeDraw']]
win_data = win_data.drop('FTR', axis=1)
win_data
# splitting data for before and after break
preCOVID = win_data.iloc[:224]
postCOVID = win_data.iloc[224:]
pd.options.display.float_format = '{:,.1f}'.format
table = win_data.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'sum', 'FTAG' : 'sum', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
away = win_data.groupby('AwayTeam').agg({'AwayTeam': 'count', 'FTAG': 'sum', 'FTHG' : 'sum', 'AwayWin' : 'sum', 'AwayDraw' : 'sum', 'AwayLoss' : 'sum'})
table = pd.concat([table,away], axis=1)
table['Points'] = table['HomeWin']*3 + table['HomeDraw']*1 + table['AwayWin']*3 + table['AwayDraw']*1
table = table.rename(columns={'HomeTeam': 'HomeGames', 'AwayTeam': 'AwayGames'})
table = table.sort_values('Points', ascending = False)
table['MP'] = table['HomeGames'] + table['AwayGames']
table['GF'] = table.iloc[:,1] + table.iloc[:,7]
table['GA'] = table.iloc[:,2] + table.iloc[:,8]
table['GD'] = table['GF'] - table['GA']
table['W'] = table['HomeWin'] + table['AwayWin']
table['D'] = table['HomeDraw'] + table['AwayDraw']
table['L'] = table['HomeLoss'] + table['AwayLoss']
table = table[['MP','W','D','L','GF','GA','GD','Points']]
table
table = win_data.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'mean', 'FTAG' : 'mean', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
away = win_data.groupby('AwayTeam').agg({'AwayTeam': 'count', 'FTAG': 'mean', 'FTHG' : 'mean', 'AwayWin' : 'sum', 'AwayDraw' : 'sum', 'AwayLoss' : 'sum'})
table = pd.concat([table,away], axis=1)
table['Points'] = table['HomeWin']*3 + table['HomeDraw']*1 + table['AwayWin']*3 + table['AwayDraw']*1
table = table.rename(columns={'HomeTeam': 'HomeGames', 'AwayTeam': 'AwayGames'})
table = table.sort_values('Points', ascending = False)
table = table.rename(columns={'FTHG': 'AveHomeGoals', 'FTAG': 'AveAwayGoals'})
table
def color_negative_red(value):
  """
  Colors elements in a dateframe
  green if positive and red if
  negative. Does not color NaN
  values.
  """

  if value < 0:
    color = 'red'
  elif value > 0:
    color = 'green'
  else:
    color = 'black'

  return 'color: %s' % color
# pre-COVID break
pd.options.display.float_format = '{:,.1f}'.format

data = preCOVID.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'mean', 'FTAG' : 'mean', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
data.columns = ['# pre-COVID matches', 'AveGF1', 'AveGA1', 'W', 'D', 'L']

data['WinRate1'] = data['W']/data['# pre-COVID matches']
data = data.sort_values('WinRate1', ascending = False)

# post-COVID break


data2 = postCOVID.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'mean', 'FTAG' : 'mean', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
data2.columns = ['# post-COVID matches', 'AveGF2', 'AveGA2', 'W2', 'D2', 'L2']

data2['WinRate2'] = data2['W2']/data2['# post-COVID matches']
# data2 = data2.sort_values('WinRate2', ascending = False)

home = pd.concat([data,data2], axis=1)
home['ΔGF'] = home['AveGF2'] - home['AveGF1']
home['ΔGA'] = home['AveGA2'] - home['AveGA1']
home['ΔWinRate'] = home['WinRate2'] - home['WinRate1']

s = home.style.applymap(color_negative_red, subset=['ΔWinRate','ΔGF','ΔGA'])
s = s.format("{:.0f}")
s = s.format({'ΔWinRate': "{:.1%}",'WinRate1': "{:.1%}", 'WinRate2': "{:.1%}",'ΔGF': "{:.1f}",'ΔGA': "{:.1f}",'AveGF1': "{:.1f}",'AveGA1': "{:.1f}",'AveGF2': "{:.1f}",'AveGA2': "{:.1f}"})
s
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].bar(['Pre-COVID', 'Post-COVID'], [home.AveGF1.mean(), home.AveGF2.mean()])
axs[0].set_title('Home Goals Scored')
axs[1].bar(['Pre-COVID', 'Post-COVID'], [home.WinRate1.mean(), home.WinRate2.mean()])
axs[1].set_title('Home Win Rate')
print('Change in Mean Home Goals: '+'{:.1f}'.format(home.ΔGF.mean()))
print('Change in Home Win Rate: '+'{:.1%}'.format(home.ΔWinRate.mean()))
# want to assess statistical significance of the change in win rate at some point

print('Pre-COVID Win Rate: '+'{:.3f}'.format(home.WinRate1.mean()))
print('Pre-COVID Win Rate sd: '+'{:.3f}'.format(home.WinRate1.std()))
print('Post-COVID Win Rate: '+'{:.3f}'.format(home.WinRate2.mean()))
print('Post-COVID Win Rate: '+'{:.3f}'.format(home.WinRate2.std()))
data = win_data.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'mean', 'FTAG' : 'mean', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
data.columns = ['# pre-COVID matches', 'AveGF1', 'AveGA1', 'W', 'D', 'L']
data['WinRate'] = data['W']/data['# pre-COVID matches']
print('Season Average Win Rate: '+'{:.3f}'.format(data.WinRate.mean()))



# pre-COVID break
pd.options.display.float_format = '{:,.1f}'.format

data3 = preCOVID.groupby('AwayTeam').agg({'AwayTeam': 'count', 'FTAG': 'mean', 'FTHG' : 'mean', 'AwayWin' : 'sum', 'AwayDraw' : 'sum', 'AwayLoss' : 'sum'})
data3.columns = ['# pre-COVID matches', 'AveGF1', 'AveGA1', 'W', 'D', 'L']

data3['WinRate1'] = data3['W']/data3['# pre-COVID matches']
data3 = data3.sort_values('WinRate1', ascending = False)

# post-COVID break
data4 = postCOVID.groupby('AwayTeam').agg({'AwayTeam': 'count', 'FTAG': 'mean', 'FTHG' : 'mean', 'AwayWin' : 'sum', 'AwayDraw' : 'sum', 'AwayLoss' : 'sum'})
data4.columns = ['# post-COVID matches', 'AveGF2', 'AveGA2', 'W2', 'D2', 'L2']

data4['WinRate2'] = data2['W2']/data2['# post-COVID matches']

home = pd.concat([data3,data4], axis=1)
home['ΔGF'] = home['AveGF2'] - home['AveGF1']
home['ΔGA'] = home['AveGA2'] - home['AveGA1']
home['ΔWinRate'] = home['WinRate2'] - home['WinRate1']

s = home.style.applymap(color_negative_red, subset=['ΔWinRate','ΔGF','ΔGA'])
s = s.format("{:.0f}")
s = s.format({'ΔWinRate': "{:.1%}",'WinRate1': "{:.1%}", 'WinRate2': "{:.1%}",'ΔGF': "{:.1f}",'ΔGA': "{:.1f}",'AveGF1': "{:.1f}",'AveGA1': "{:.1f}",'AveGF2': "{:.1f}",'AveGA2': "{:.1f}"})
s
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].bar(['Pre-COVID', 'Post-COVID'], [home.AveGF1.mean(), home.AveGF2.mean()])
axs[0].set_title('Away Goals Scored')
axs[1].bar(['Pre-COVID', 'Post-COVID'], [home.WinRate1.mean(), home.WinRate2.mean()])
axs[1].set_title('Away Win Rate')
print('Change in Mean Away Goals: '+'{:.1f}'.format(home.ΔGF.mean()))
print('Change in Away Win Rate: '+'{:.1%}'.format(home.ΔWinRate.mean()))
table = preCOVID.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'sum', 'FTAG' : 'sum', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
away = preCOVID.groupby('AwayTeam').agg({'AwayTeam': 'count', 'FTAG': 'sum', 'FTHG' : 'sum', 'AwayWin' : 'sum', 'AwayDraw' : 'sum', 'AwayLoss' : 'sum'})
table = pd.concat([table,away], axis=1)
table['Points'] = table['HomeWin']*3 + table['HomeDraw']*1 + table['AwayWin']*3 + table['AwayDraw']*1
table = table.rename(columns={'HomeTeam': 'HomeGames', 'AwayTeam': 'AwayGames'})
table = table.sort_values('Points', ascending = False)
table['MP'] = table['HomeGames'] + table['AwayGames']
table['GF'] = table.iloc[:,1] + table.iloc[:,7]
table['GA'] = table.iloc[:,2] + table.iloc[:,8]
table['GD'] = table['GF'] - table['GA']
table['W'] = table['HomeWin'] + table['AwayWin']
table['D'] = table['HomeDraw'] + table['AwayDraw']
table['L'] = table['HomeLoss'] + table['AwayLoss']
table = table[['MP','W','D','L','GF','GA','GD','Points']]

table1 = postCOVID.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'sum', 'FTAG' : 'sum', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
away = postCOVID.groupby('AwayTeam').agg({'AwayTeam': 'count', 'FTAG': 'sum', 'FTHG' : 'sum', 'AwayWin' : 'sum', 'AwayDraw' : 'sum', 'AwayLoss' : 'sum'})
table1 = pd.concat([table1,away], axis=1)
table1['Points'] = table1['HomeWin']*3 + table1['HomeDraw']*1 + table1['AwayWin']*3 + table1['AwayDraw']*1
table1 = table1.rename(columns={'HomeTeam': 'HomeGames', 'AwayTeam': 'AwayGames'})
table1 = table1.sort_values('Points', ascending = False)
table1['MP'] = table1['HomeGames'] + table1['AwayGames']
table1['GF'] = table1.iloc[:,1] + table1.iloc[:,7]
table1['GA'] = table1.iloc[:,2] + table1.iloc[:,8]
table1['GD'] = table1['GF'] - table1['GA']
table1['W'] = table1['HomeWin'] + table1['AwayWin']
table1['D'] = table1['HomeDraw'] + table1['AwayDraw']
table1['L'] = table1['HomeLoss'] + table1['AwayLoss']
table1 = table1[['MP','W','D','L','GF','GA','GD','Points']]

table = pd.concat([table,table1], axis=1)
table
