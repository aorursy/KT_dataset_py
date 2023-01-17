import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from matplotlib import style

import time

from subprocess import check_output

from functools import reduce



start_time = time.time()



###Research question #1:###

###Team SoloMid or also known as TSM has been very consistent with domestic competitions like the

###NALCS. The fans' hope for TSM when it comes to international stage is very high. However, TSM 

###always fail to achieve the critics' and fans' expectations. 

###Why is TSM not dominant in the International stage?





###Get all the matches from 2016###

###Get TSM's NALCS matches###

games = pd.read_csv('../input/_LeagueofLegends.csv')

df_NALCS = pd.DataFrame(games)

df_NALCS = df_NALCS[(df_NALCS['League'] == 'North_America') & (df_NALCS['Year'] == 2016) & 

                    ((df_NALCS['blueTeamTag'] == 'TSM') | (df_NALCS['redTeamTag'] == 'TSM'))]

NALCS_games = df_NALCS[['MatchHistory','League','Year','blueTeamTag','redTeamTag','gamelength']]



###Get TSM's Worlds matches###

df_Worlds = pd.DataFrame(games)

df_Worlds = df_Worlds[(df_Worlds['League'] == 'Season_World_Championship') & (df_Worlds['Year'] == 2016) & 

                    ((df_Worlds['blueTeamTag'] == 'TSM') | (df_Worlds['redTeamTag'] == 'TSM'))]

Worlds_games = df_Worlds[['MatchHistory','League','Year','blueTeamTag','redTeamTag','gamelength']]



###Get the gold values of each TSM's matches###

gold_tsm = pd.read_csv('../input/goldValues.csv')

df_gold_tsm = pd.DataFrame(gold_tsm)

tsm_stats_NALCS = pd.merge(df_gold_tsm,NALCS_games,on='MatchHistory').fillna(0)

tsm_stats_Worlds = pd.merge(df_gold_tsm,Worlds_games,on='MatchHistory').fillna(0)

tsm_stats_NALCS = tsm_stats_NALCS[tsm_stats_NALCS['NameType'] != 'golddiff']

tsm_stats_Worlds = tsm_stats_Worlds[tsm_stats_Worlds['NameType'] != 'golddiff']



###TSM's average game time during NALCS and Worlds

NALCS_ave_gametime = NALCS_games[['gamelength']].mean()

Worlds_ave_gametime = Worlds_games[['gamelength']].mean()

print('TSM average game time during the NALCS: {:.2f}mins'.format(NALCS_ave_gametime['gamelength']))

print('TSM average game time during the Worlds: {:.2f}mins'.format(Worlds_ave_gametime['gamelength']))

print('TSM have longer games at Worlds compared to the NALCS')



###average gold difference###

NALCS_team_gold = tsm_stats_NALCS[(tsm_stats_NALCS['NameType'] == 'goldblue') | 

                                  (tsm_stats_NALCS['NameType'] == 'goldred')]

Worlds_team_gold = tsm_stats_Worlds[(tsm_stats_Worlds['NameType'] == 'goldblue') | 

                                  (tsm_stats_Worlds['NameType'] == 'goldred')]

NALCS_team_gold = NALCS_team_gold.drop(['Year','gamelength'],axis=1)

Worlds_team_gold = Worlds_team_gold.drop(['Year','gamelength'],axis=1)

NALCS_team_gold = NALCS_team_gold.mean()

Worlds_team_gold = Worlds_team_gold.mean()

NALCS_team_gold.plot(color='red')

Worlds_team_gold.plot(color='blue')

NALCS_patch = mpatches.Patch(color='red',label='NALCS')

Worlds_patch = mpatches.Patch(color='blue',label='Worlds')

plt.legend(handles=[NALCS_patch,Worlds_patch])

plt.show()

print('The graph shows that TSM manages to keep the gold difference in their favor')

print('But, at some point they will lose the lead')



###KDA Analysis###

kda_tsm = pd.read_csv('../input/deathValues.csv')

df_kda_tsm = pd.DataFrame(kda_tsm)

NALCS_team_kda = pd.merge(df_kda_tsm,NALCS_games,on='MatchHistory').fillna('')

Worlds_team_kda = pd.merge(df_kda_tsm,Worlds_games,on='MatchHistory').fillna('')

NALCS_team_kda = NALCS_team_kda[['Victim','Killer','Assist_1','Assist_2','Assist_3','Assist_4']]

Worlds_team_kda = Worlds_team_kda[['Victim','Killer','Assist_1','Assist_2','Assist_3','Assist_4']]

NALCS_tsm_kills = len(list(filter(lambda x: 'TSM' in x,NALCS_team_kda['Killer'])))

NALCS_tsm_deaths = len(list(filter(lambda x: 'TSM' in x,NALCS_team_kda['Victim'])))

NALCS_tsm_assists = len(list(filter(lambda x: 'TSM' in x,NALCS_team_kda['Assist_1'])))

NALCS_tsm_assists += len(list(filter(lambda x: 'TSM' in x,NALCS_team_kda['Assist_2'])))

NALCS_tsm_assists += len(list(filter(lambda x: 'TSM' in x,NALCS_team_kda['Assist_3'])))

NALCS_tsm_assists += len(list(filter(lambda x: 'TSM' in x,NALCS_team_kda['Assist_4'])))

Worlds_tsm_kills = len(list(filter(lambda x: 'TSM' in x,Worlds_team_kda['Killer'])))

Worlds_tsm_deaths = len(list(filter(lambda x: 'TSM' in x,Worlds_team_kda['Victim'])))

Worlds_tsm_assists = len(list(filter(lambda x: 'TSM' in x,Worlds_team_kda['Assist_1'])))

Worlds_tsm_assists += len(list(filter(lambda x: 'TSM' in x,Worlds_team_kda['Assist_2'])))

Worlds_tsm_assists += len(list(filter(lambda x: 'TSM' in x,Worlds_team_kda['Assist_3'])))

Worlds_tsm_assists += len(list(filter(lambda x: 'TSM' in x,Worlds_team_kda['Assist_4'])))

print('TSMs NALCS KDA is {:.2f}'.format((NALCS_tsm_kills+NALCS_tsm_assists)/NALCS_tsm_deaths))

print('TSMs Worlds KDA is {:.2f}'.format((Worlds_tsm_kills+Worlds_tsm_assists)/Worlds_tsm_deaths))

print('TSMs KDA at Worlds is lower compared to their ratio at the NALCS')

print('This means that the teams during the World Championship are more aggressive compared to the NALCS teams')



print('Total run time: {}sec'.format(time.time() - start_time))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import matplotlib.pyplot as plt

from matplotlib import style

import time

from subprocess import check_output

import operator



start_time = time.time()

###Does KDA win games?###



###KDA ratio of players based on 2016 games###

games = pd.read_csv('../input/_LeagueofLegends.csv')

kda = pd.read_csv('../input/deathValues.csv')

df_games = pd.DataFrame(games)

df_games = df_games[df_games['Year'] == 2016]

df_kda = pd.DataFrame(kda).fillna(0)

df_kda = pd.merge(df_games,df_kda,on='MatchHistory')



pro_players = df_kda[(df_kda['League'] == 'North_America') |

                      (df_kda['League'] == 'Europe')|

                      (df_kda['League'] == 'LCK')|

                      (df_kda['League'] == 'LMS')]['Killer'].unique()



pro_players_kills = {player: float(((df_kda[df_kda['Killer'] == player]['Killer'].value_counts() +

                     df_kda[df_kda['Assist_1'] == player]['Assist_1'].value_counts()) /

                     df_kda[df_kda['Victim'] == player]['Victim'].value_counts()).fillna(0))

                     for player in pro_players}

top_players = dict(sorted(pro_players_kills.items(), key=operator.itemgetter(1), reverse=True)[:11])

del top_players[0]

print('Players with the best KDA internationally:')

for key, value in top_players.items():

    print('{} {:.2f}'.format(key,value))

print('Compared to the KDA of the 2016 World Champions')

for key, value in pro_players_kills.items():

    if 'SKT ' in str(key):

        print('{} {:.2f}'.format(key,value))

print('They have a low KDA but they managed to win the 2016 World Championship')

print('Total run time is {}'.format(time.time()-start_time))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import matplotlib.pyplot as plt

from matplotlib import style

import time

from subprocess import check_output

import matplotlib.patches as mpatches



start_time = time.time()

###Regional difference###

###Difference between the statistics of major regions###



games = pd.read_csv('../input/_LeagueofLegends.csv')

kda = pd.read_csv('../input/deathValues.csv')

gold_diff = pd.read_csv('../input/goldValues.csv')



df_games = pd.DataFrame(games)

df_kda = pd.DataFrame(kda)

df_gold_diff = pd.DataFrame(gold_diff)



df_games = df_games[df_games['Year'] == 2016]

df_kda = pd.merge(df_kda,df_games,on='MatchHistory').fillna(0)

df_gold_diff = pd.merge(df_gold_diff,df_games,on='MatchHistory').fillna(0)



NA_games = df_games[df_games['League'] == 'North_America']

EU_games = df_games[df_games['League'] == 'Europe']

LCK_games = df_games[df_games['League'] == 'LCK']

LMS_games = df_games[df_games['League'] == 'LMS']



ave_game_time_NA = NA_games[['gamelength']].mean()

ave_game_time_EU = EU_games[['gamelength']].mean()

ave_game_time_LCK = LCK_games[['gamelength']].mean()

ave_game_time_LMS = LMS_games[['gamelength']].mean()



print('Average game time in North American League is {:.2f} mins'.format(float(ave_game_time_NA)))

print('Average game time in European League is {:.2f} mins'.format(float(ave_game_time_EU)))

print('Average game time in Korean League is {:.2f} mins'.format(float(ave_game_time_LCK)))

print('Average game time in Taiwanese League is {:.2f} mins'.format(float(ave_game_time_LMS)))





NA_patch = mpatches.Patch(color='red',label='NA')

EU_patch = mpatches.Patch(color='blue',label='EU')

LCK_patch = mpatches.Patch(color='green',label='LCK')

LMS_patch = mpatches.Patch(color='black',label='LMS')

gold_diff_NA = df_gold_diff[df_gold_diff['League'] == 'North_America'].drop(df_games,axis=1).mean().plot(color='red')

gold_diff_EU = df_gold_diff[df_gold_diff['League'] == 'Europe'].drop(df_games,axis=1).mean().plot(color='blue')

gold_diff_LCK = df_gold_diff[df_gold_diff['League'] == 'LCK'].drop(df_games,axis=1).mean().plot(color='green')

gold_diff_LMS = df_gold_diff[df_gold_diff['League'] == 'LMS'].drop(df_games,axis=1).mean().plot(color='black')

plt.legend(handles=[NA_patch,EU_patch,LCK_patch,LMS_patch])

plt.show()

print('LCK has the highest average gold difference')





print('Total run time is {}'.format(time.time()-start_time))