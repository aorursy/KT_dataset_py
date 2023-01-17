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
game_data = pd.read_csv(r'/kaggle/input/european-football-database-20192020/E0.csv')
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
preCOVID = win_data.iloc[:288]
postCOVID = win_data.iloc[288:]
# as of 02/07/2020

table = win_data.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'mean', 'FTAG' : 'mean', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
away = win_data.groupby('AwayTeam').agg({'AwayTeam': 'count', 'FTAG': 'mean', 'FTHG' : 'mean', 'AwayWin' : 'sum', 'AwayDraw' : 'sum', 'AwayLoss' : 'sum'})
table = pd.concat([table,away], axis=1)
table['Points'] = table['HomeWin']*3 + table['HomeDraw']*1 + table['AwayWin']*3 + table['AwayDraw']*1
table = table.sort_values('Points', ascending = False)
table
# pre-COVID break
pd.options.display.float_format = '{:,.1f}'.format

data = preCOVID.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'mean', 'FTAG' : 'mean', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
data.columns = ['# pre-COVID matches', 'Ave. Goals For', 'Ave. Goals Against', 'Won', 'Drawn', 'Lost']

data['WinRate'] = data['Won']/data['# pre-COVID matches'] * 100
data = data.sort_values('WinRate', ascending = False)

# post-COVID break
pd.options.display.float_format = '{:,.1f}'.format

data2 = postCOVID.groupby('HomeTeam').agg({'HomeTeam': 'count', 'FTHG': 'mean', 'FTAG' : 'mean', 'HomeWin' : 'sum', 'HomeDraw' : 'sum', 'HomeLoss' : 'sum'})
data2.columns = ['# post-COVID matches', 'Ave. Goals For', 'Ave. Goals Against', 'Won', 'Drawn', 'Lost']

data2['WinRate'] = data2['Won']/data2['# post-COVID matches'] * 100
data2 = data2.sort_values('WinRate', ascending = False)

home = pd.concat([data,data2], axis=1)
home
# checking the Liverpool result because I thought something went wrong - they haven't lost a single home game!
liverpool = win_data[win_data['HomeTeam'] == 'Liverpool']
liverpool
# pre-COVID

pd.options.display.float_format = '{:,.1f}'.format

data3 = preCOVID.groupby('AwayTeam').agg({'AwayTeam': 'count', 'FTAG': 'mean', 'FTHG' : 'mean', 'AwayWin' : 'sum', 'AwayDraw' : 'sum', 'AwayLoss' : 'sum'})
data3.columns = ['# pre-COVID matches', 'Ave. Goals For', 'Ave. Goals Against', 'Won', 'Drawn', 'Lost']

data3['WinRate'] = data3['Won']/data3['# pre-COVID matches'] * 100
data3 = data3.sort_values('WinRate', ascending = False)

# post-COVID break

data4 = postCOVID.groupby('AwayTeam').agg({'AwayTeam': 'count', 'FTAG': 'mean', 'FTHG' : 'mean', 'AwayWin' : 'sum', 'AwayDraw' : 'sum', 'AwayLoss' : 'sum'})
data4.columns = ['# post-COVID matches', 'Ave. Goals For', 'Ave. Goals Against', 'Won', 'Drawn', 'Lost']

data4['WinRate'] = data4['Won']/data4['# post-COVID matches'] * 100
data4 = data4.sort_values('WinRate', ascending = False)

away = pd.concat([data3,data4], axis=1)
away
