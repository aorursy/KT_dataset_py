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
pd.set_option('display.max_columns', None)
pitching=pd.read_csv('/kaggle/input/baseball-databank/Pitching.csv')
pitching
pitching.columns
career_pitching=pd.DataFrame()
for player in pitching['playerID'].value_counts().index:
    df=pitching[pitching['playerID'].isin([player])]
    career_pitching=career_pitching.append({'playerID':player,'Seasons':len(df)},ignore_index=True)
#players much play at least 10 seasons to be eligible for the hall of fame
minimum=career_pitching[career_pitching['Seasons']>=10]
minimum
df=pitching[pitching['playerID'].isin(['ryanno01'])].drop(columns=['playerID','lgID','yearID','teamID'])
df['IP']=df['IPouts']/3
for column in df:
    if column=='BAOpp':
        print(column)
pitching_final=pd.DataFrame()
for player in minimum['playerID']:
    player_totals=pd.DataFrame()
    player_totals=player_totals.append({'playerID':player},ignore_index=True)
    df=pitching[pitching['playerID'].isin([player])].drop(columns=['playerID','lgID','yearID','teamID'])
    df['IP']=df['IPouts']/3
    for column in df:
        if column=='BAOpp':
            player_totals[column]=df.mean()[column]
        elif column=='ERA':
            player_totals[column]=(9*df.sum()['ER'])/(df.sum()['IP'])
        else:
            player_totals[column]=df.sum()[column]
    pitching_final=pitching_final.append(player_totals)
pitching_final=pitching_final.reset_index().drop(columns=['IPouts','index'])
pitching_final
pitching_final.to_csv('pitching.csv')