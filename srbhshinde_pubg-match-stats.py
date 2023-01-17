# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#set max display column values
pd.set_option('display.max_columns', 100)
# read dataset
pubg = pd.read_csv('../input/train.csv')
pubg.head(5)
# plotting distribution for number of groups
plt.figure(figsize=(10,5))
sns.distplot(pubg['numGroups'], bins=30)
# adding an extra column: Match Type
def match_type(x):
    if x <= 25:
        return 'Squad'
    elif (x > 25) & (x <= 50):
        return 'Duo'
    else:
        return 'Solo'

pubg['matchType'] = pubg['numGroups'].apply(match_type)
# function to display match stats
def match_stats(x):
    print('\n')
    t = pubg[pubg['matchId'] == x]['matchType'].iloc[0]
    print('***** Match Type:', t + ' *****')
    print('Match ID: ', x)
    
    print('Max Assists:', pubg[pubg['matchId'] == x]['assists'].max())
    
    print('Max Boosts Used:', pubg[pubg['matchId'] == x]['boosts'].max())
    
    print('Max Damage Dealt: {:.3f}'.format(pubg[pubg['matchId'] == x].groupby('groupId').sum()['damageDealt'].max()))
    
    print('Max Damage Dealt by Winning Group: {:.3f}'.format(pubg[(pubg['matchId'] == x) & (pubg['winPlacePerc'] == 1)].groupby('groupId').sum()['damageDealt'].iloc[0]))
    
    if t == 'Solo':
        print('Max Knockouts: 0')
    else:
        print('Max Knockouts:', pubg[pubg['matchId'] == x].groupby('groupId').sum()['DBNOs'].max())
    
    print('Max Headshot Kills by a Group:', pubg[pubg['matchId'] == x].groupby('groupId').sum()['headshotKills'].max())
    return                                                                                                                      
#for single match
match_stats(10)
#for multiple matches
m_id = [23, 786, 13721, 38273]
for i in m_id:
    match_stats(i)