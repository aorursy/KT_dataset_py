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
batting=pd.read_csv('/kaggle/input/the-history-of-baseball/batting.csv')
batting
batting=batting.fillna(0)
batting.columns
batting['player_id'].value_counts()
pd.set_option('display.max_columns', None)
batting.head(1)
#get rid of team_id column, because there will be no career total for this
batting=batting.drop(columns=['team_id'])
#see where .sum() error is from
print(batting['player_id'].dtypes)
df=batting[batting['player_id'].isin(['johnto01'])].drop(columns=['player_id'])
for column in df:
    print(df[column].dtypes)
career_batting=pd.DataFrame()
for player in batting['player_id'].value_counts().index:
    df=batting[batting['player_id'].isin([player])]
    career_batting=career_batting.append({'player_id':player,'Seasons':len(df)},ignore_index=True)
#players much play at least 10 seasons to be eligible for the hall of fame
minimum=career_batting[career_batting['Seasons']>=10]
minimum
#check methodology for getting a player's career totals
batting_final=pd.DataFrame()
batting_final=batting_final.append({'player_id':'pasquda01'},ignore_index=True)
df=batting[batting['player_id'].isin(['pasquda01'])].drop(columns=['player_id','league_id','year'])
for column in df:
    batting_final[column]=df.sum()[column]
batting_final
batting_final=pd.DataFrame()
for player in minimum['player_id']:
    player_totals=pd.DataFrame()
    player_totals=player_totals.append({'player_id':player},ignore_index=True)
    df=batting[batting['player_id'].isin([player])].drop(columns=['player_id','league_id','year'])
    for column in df:
        player_totals[column]=df.sum()[column]
    batting_final=batting_final.append(player_totals)
#all indices are zero, so fix this by resetting index and getting rid of oroginal indices
batting_final=batting_final.reset_index().drop(columns='index')
batting_final
batting_final.to_csv('batting.csv')