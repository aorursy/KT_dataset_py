import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

from pprint import pprint
#reading the ipl dataset into a dataframe

ipl_matches = pd.read_csv('../input/matches.csv')

print(type(ipl_matches))



#dimensions of the data

ipl_matches.shape
#attributes of the dataset (columns)

print(ipl_matches.columns.tolist())



#matches won by Mumbai Indians batting in the second innings

mi_sec_inn = ipl_matches.loc[ (ipl_matches['win_by_runs'] > 0)

               & (ipl_matches['result']=='normal') 

               & ((ipl_matches['team1']=='Mumbai Indians') | (ipl_matches['team2']=='Mumbai Indians'))]

mi_sec_inn.head()
#matches won by MI at home

mi_home_wins = pd.DataFrame(ipl_matches.loc[ (ipl_matches['winner'] == 'Mumbai Indians')

               & (ipl_matches['result']=='normal') 

               & (ipl_matches['team1']=='Mumbai Indians')].groupby('season')['winner'].count())



barWidth = 0.5

xh = mi_home_wins.index.tolist()

yh = list(mi_home_wins['winner'])



mi_away_wins = pd.DataFrame(ipl_matches.loc[ (ipl_matches['winner'] == 'Mumbai Indians')

               & (ipl_matches['result']=='normal') 

               & (ipl_matches['team1']=='Mumbai Indians')].groupby('season')['winner'].count())



xa = [i+barWidth for i in xh]

ya = list(mi_away_wins['winner'])



plt.bar(xh, yh,width=barWidth,color='blue',edgecolor='white',label='Home Matches won')

plt.bar(xa, ya,width=barWidth,color='red',edgecolor='white',label='Away matches won')

plt.legend()

plt.show()