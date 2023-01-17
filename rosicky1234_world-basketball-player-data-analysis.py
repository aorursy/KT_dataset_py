#https://www.kaggle.com/jacobbaruch/basketball-players-stats-per-season-49-leagues/notebooks



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df=pd.read_csv("/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv")

df.head()
df.columns
all_league = df['League'].unique()

print(all_league)

df['MinPerGame'] = round(df['MIN']/df['GP'],1)

df['PtsPerGame'] = round(df['PTS']/df['GP'],1)

df['RebPerGame']=round(df['REB']/df['GP'],1)

df['AstPerGame']=round(df['AST']/df['GP'],1)

df['BlkPerGame']=round(df['BLK']/df['GP'],1)

df['PfPerGame']=round(df['PF']/df['GP'],1)

df['ToPerGame']=round(df['TOV']/df['GP'],1)

df['StlPerGame']=round(df['STL']/df['GP'],1)



df['FieldGoal%']=round(df['FGM']*100/df['FGA'],1)

df['3Pts%']=round(df['3PM']*100/df['3PA'],1)

df['FT%']=round(df['FTM']*100/df['FTA'],1)



df['AstTovRatio'] = round(df['AST']/df['TOV'],1)

df['BMI'] = (df['weight']/2.2) / pow(df['height_cm']/100,2)

df['age']=2020-df['birth_year']



df.head()
NBA_players = df[df['League']=='NBA']

Other_players = df[df['League']!='NBA']





NBA_Regular = NBA_players[ (NBA_players['Season']=='2018 - 2019') & (NBA_players['Stage']=='Regular_Season')]



y_value = ['PtsPerGame','AstPerGame','RebPerGame','AstTovRatio']



for y in y_value:

    temp = NBA_Regular.nlargest(15,y)

    ax = temp.plot.bar(x='Player', y=y)

    title = 'Top 15 ' + y + ' players in Regular Season' 

    plt.title(title)

#Players in 50-40-90 Club 





NBA_504090 = NBA_Regular[(NBA_Regular['FieldGoal%']>=50)&(NBA_Regular['3PM']>=40)&(NBA_Regular['FT%']>=90)]



print(NBA_504090[['Player','Team']])
NBA_504080 = NBA_Regular[(NBA_Regular['FieldGoal%']>=50)&(NBA_Regular['3PM']>=40)&(NBA_Regular['FT%']>=80)]



print(NBA_504080[['Player','Team']])
NBA_Playoff = NBA_players[ (NBA_players['Season']=='2018 - 2019') & (NBA_players['Stage']=='Playoffs')]

NBA_Playoff.head()





y_value = ['PtsPerGame','AstPerGame','RebPerGame','AstTovRatio']



for y in y_value:

    temp = NBA_Playoff.nlargest(15,y)

    ax = temp.plot.bar(x='Player', y=y)

    title = 'Top 15 ' + y + ' players in Playoffs' 

    plt.title(title)
CJ = NBA_Playoff[NBA_Playoff['Player']=='Cory Joseph']

CJ = CJ[['Player','ToPerGame']]

print(CJ)
Other_players_1920 = Other_players[Other_players['Season']=='2019 - 2020']

Young_players_1920 = Other_players[Other_players['age']<= 25]



print('Potential good Playmakers'+'\n')



Gd_Playmaker = Young_players_1920[ (Young_players_1920['AstPerGame']>=6) & (Young_players_1920['AstTovRatio']>=3)]

print(Gd_Playmaker[['League','Player','AstPerGame','AstTovRatio']])
GD_scorer =  Young_players_1920[ (Young_players_1920['PtsPerGame']>=20) & (Young_players_1920['FieldGoal%']>=45) & (Young_players_1920['3Pts%']>=40)]

GD_scorer.reset_index(drop=True, inplace=True)

print('Potential Scorers'+'\n')

print(GD_scorer[['League','Player','PtsPerGame','FieldGoal%','3Pts%']])
GD_insider =  Young_players_1920[ (Young_players_1920['RebPerGame']>=10) & (Young_players_1920['BlkPerGame']>=1) ]

print('Potential Inside Man'+'\n')

print(GD_insider[['League','Player','RebPerGame','BlkPerGame']])
GD_allRound =  Young_players_1920[ (Young_players_1920['PtsPerGame']>=15) &(Young_players_1920['RebPerGame']>=5) & (Young_players_1920['AstPerGame']>=5) ]

print('Potential All-Rounded Player'+'\n')

print(GD_allRound[['League','Player','PtsPerGame','AstPerGame','RebPerGame']])