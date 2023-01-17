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
batting=pd.read_csv('/kaggle/input/career-hitting/batting.csv').drop(columns='Unnamed: 0')
batting
master=pd.read_excel('/kaggle/input/correct/correct.xlsx')
master=master[['IDPLAYER','PLAYERNAME']]
master
master['IDPLAYER'].str.contains('abreubo01').any()
players=[]
for player in batting['player_id']:
    if master['IDPLAYER'].str.contains(player).any():
        players.append(player)
len(players)
len(batting)
hof=pd.read_csv('/kaggle/input/baseball-databank/HallOfFame.csv')
hof
#df of all players included in in HOF dataset
ballot=batting[batting['player_id'].isin(hof['playerID'].tolist())]
ballot
#df of all players not in HOF dataset
absent=batting[-batting['player_id'].isin(hof['playerID'].tolist())]
absent
absent.sort_values(by='rbi',ascending=False)
hits=absent[absent['h']>=2000]
hits
len(hits)
hof[hof['playerID'].isin(batting['player_id'].tolist())]
hof.columns
hof['category'].value_counts()
hof['needed_note'].value_counts()
hof=hof[hof['category'].isin(['Player'])]
hof=hof.drop(columns=['needed_note'])
hof
hof['percent']=hof['votes']/hof['ballots']
hof['threshold']=hof['needed']/hof['ballots']
hof[hof['inducted'].isin(['Y'])]
type(hof['inducted'][0])
if hof['inducted'].str.contains('Y').any():
    print('yes')
df=hof[hof['playerID'].isin(['youngro01'])]
if df['inducted'].str.contains('Y').any():
    inducted=df[df['inducted'].isin(['Y'])].reset_index()
    print(inducted['percent'][0])
#     threshold=inducted['threshold'][0]
#how to check if there are any null values in column
#df['your column name'].isnull().values.any()
hof_clean=pd.DataFrame()
for player in hof['playerID'].value_counts().index:
    playerdf=pd.DataFrame()
    
    #extract player data
    df=hof[hof['playerID'].isin([player])].reset_index()
        
    #store information to append
#     induction=df[df['inducted'].isin(['Y'])].reset_index()
    year=df.max()['yearid']
    #voter=df['votedBy'][0]
    
#     df['percent']=df['votes']/df['ballots']
#     df['threshold']=df['needed']/df['ballots']
    
#     ballots=df.sum()['ballots']/len(df)
#     needed=df.sum()['needed']/len(df)
#     votes=df.sum()['votes']/len(df)
    attempts=len(df)
    if df['inducted'].str.contains('Y').any():
        inducted=df[df['inducted'].isin(['Y'])].reset_index()
        percent=inducted['percent'][0]
        threshold=inducted['threshold'][0]
        votedBy=inducted['votedBy'][0]
        playerdf=playerdf.append({'year':year,'player_id':player,'percent':percent,'threshold':
                                threshold,'years':attempts,'votedBy':votedBy,'inducted':'Y'},ignore_index=True)
    else:
#         year=df.max()['yearid']
#         voter=df['votedBy'].value_counts().index[0]
#         ballots=df.sum()['ballots']/len(df)
#         needed=df.sum()['needed']/len(df)
#         votes=df.sum()['votes']/len(df)
#         attempts=len(df)
        max_percent=df.max()['percent']
        threshold=df['threshold'][df['percent'].idxmax()]
        votedBy=df['votedBy'].value_counts().index[0]
        playerdf=playerdf.append({'year':year,'player_id':player,'percent':max_percent,'threshold':
                                threshold,'years':attempts,'votedBy':votedBy,'inducted':'N'},ignore_index=True)
    hof_clean=hof_clean.append(playerdf)
hof_clean=hof_clean.reset_index().drop(columns='index')
hof_clean
hof_clean[-hof_clean['votedBy'].isin(['BBWAA'])]['threshold'].isnull().value_counts()
combined=batting.join(hof_clean.set_index('player_id'), on='player_id')
combined
#take out players that aren't in hof dataset
clean=combined[-combined['inducted'].isnull()]
clean
clean['inducted'].value_counts()
clean[clean['votedBy'].isin(['BBWAA'])&clean['inducted'].isin(['Y'])].sort_values(by='h')
pitchers=pd.read_csv('/kaggle/input/baseball-databank/Pitching.csv')
pitchers
pitchers_list=[]
for player in clean['player_id']:
    if pitchers['playerID'].str.contains(player).any():
        df=clean[clean['player_id'].isin([player])].reset_index()
        
        #weed out hitters that still show up, if they have less than 1000 hits then they are not in the
        #hof because of their hitting statistics
        if df['h'][0] <1000:
            pitchers_list.append(player)
pitchers_list
hitters=clean[-clean['player_id'].isin(pitchers_list)]
hitters
hitters['inducted'].value_counts()
hitters.sort_values(by='hr',ascending=False).head(10)
#reset indices
hitters=hitters.reset_index().drop(columns='index')
hitters.to_csv('hitters2.csv')