# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train_V2.csv')
data.iloc[:10,]
print('There are {:,} rows and {} columns.'.format(data.shape[0], data.shape[1]))
print('There are {} continuous variables and {} categorical variables.'.format(len([x for x in data.dtypes if x == 'int64' or x == 'float64']),len([x for x in data.dtypes if x == 'object'])))
cat_var = [data.columns[i] for i, x in enumerate(data.dtypes) if x == 'object']
print('The categorical variables are {}.'.format(cat_var))
cont_var = [data.columns[i] for i, x in enumerate(data.dtypes) if x == 'int64' or x == 'float64']
print('\nThe continuous variables are {}.'.format(cont_var))
print('Id - Player’s Id')
Id_describe = data.Id.describe()
print(Id_describe)
print('groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.')
groupId_describe = data.groupId.describe()
print(groupId_describe)
print('{:.2%} of groupId is unique.'.format(groupId_describe['unique']/groupId_describe['count']))
groupId_freq = data.groupId.value_counts()
groupId_freq.plot(kind='box')
print(groupId_freq.describe())
print('The median frequency of groupId is {}.'.format(groupId_freq.median()))
print('matchId - ID to identify match. There are no matches that are in both the training and testing set.')
matchId_describe = data.matchId.describe()
print(matchId_describe)
print('{:.2%} of matchId is unique.'.format(matchId_describe['unique']/matchId_describe['count']))
matchId_freq = data.matchId.value_counts()
matchId_freq.plot(kind='box')
print(matchId_freq.describe())
print('The median frequency of matchId is {}.'.format(matchId_freq.median()))
print('matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.')
matchType_describe = data.matchType.describe()
print(matchType_describe)

matchType_freq = data.matchType.value_counts()
matchType_freq.plot(kind='box')
print(matchType_freq.describe())
matchType_freq.plot(kind='bar', title='Frequency plot of matchType')
print('assists - Number of enemy players this player damaged that were killed by teammates.')
assists_describe = data.assists.describe()
print(assists_describe)

data.assists.plot(kind='box')
print('The median assists is {}.'.format(data.assists.median()))
print('There are {:,} observations with at least one assists, which is {:.2%} of total observations.'.format(len([x for x in data.assists if x > 0]),len([x for x in data.assists if x > 0])/len(data.assists)))
print('boosts - Number of boost items used.')
print("Boosts items are items that increases the player's boost bar, which benefits the player's health regeneration and/or movement speed.")
boosts_describe = data.boosts.describe()
print(boosts_describe)

data.boosts.plot(kind='box')
print('The median boost item used is {}.'.format(data.boosts.median()))
print('There are {:,} observations with at least one boosts, which is {:.2%} of total observations.'.format(len([x for x in data.boosts if x > 0]),len([x for x in data.boosts if x > 0])/len(data.boosts)))
print('damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.')

damageDealt_describe = data.damageDealt.describe()
print(damageDealt_describe)

data.damageDealt.plot(kind='box')
print('The median damageDealt is {}.'.format(data.damageDealt.median()))
print('There are {:,} observations with damageDealth more than zero, which is {:.2%} of total observations.'.format(len([x for x in data.damageDealt if x > 0]),len([x for x in data.damageDealt if x > 0])/len(data.damageDealt)))
print('DBNOs - Number of enemy players knocked.')
print('''Knocked down happens when a player hit points reach zero during a duo or squad match. 
In solo games, player dies instead of knocked down.
Knocking enemy down is not equivalent to killing the enemy.''')
DBNOs_describe = data.DBNOs.describe()
print(DBNOs_describe)

data.DBNOs.plot(kind='box')
print('The median DBNOs is {}.'.format(data.DBNOs.median()))
print('There are {:,} observations with DBNOs is at least one, which is {:.2%} of total observations.'.format(len([x for x in data.DBNOs if x > 0]),len([x for x in data.DBNOs if x > 0])/len(data.DBNOs)))
print('headshotKills - Number of enemy players killed with headshots')

headshotKills_describe = data.headshotKills.describe()
print(headshotKills_describe)

data.headshotKills.plot(kind='box')
print('The median headshotKills is {}.'.format(data.headshotKills.median()))
print('There are {:,} observations with headshotKills is at least one, which is {:.2%} of total observations.'.format(len([x for x in data.headshotKills if x > 0]),len([x for x in data.headshotKills if x > 0])/len(data.headshotKills)))
print('heals - Number of healing items used.')
print("Healing items are items that increases the player's health.")
heals_describe = data.heals.describe()
print(heals_describe)

data.heals.plot(kind='box')
print('The median healing items used is {}.'.format(data.heals.median()))
print('There are {:,} observations with heals is at least one, which is {:.2%} of total observations.'.format(len([x for x in data.heals if x > 0]),len([x for x in data.heals if x > 0])/len(data.heals)))
print('killPlace - Ranking in match of number of enemy players killed.')
print('killPlace is the ranking in terms of number of enemy killed')
killPlace_describe = data.killPlace.describe()
print(killPlace_describe)

data.killPlace.plot(kind='box')
print('killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.')

killPoints_describe = data.killPoints.describe()
print(killPoints_describe)

data.killPoints.plot(kind='hist')

data.killPoints.plot(kind='box')
print('There are {:,} observations with killPoints equals zero, which is {:.2%} of total observations.'.format(len([x for x in data.killPoints if x == 0]),len([x for x in data.killPoints if x == 0])/len(data.killPoints)))
print('The lowest rank other than zero is {}.'.format(min([x for x in data.killPoints if x != 0])))
print('kills - Number of enemy players killed.')

kills_describe = data.kills.describe()
print(kills_describe)

data.kills.plot(kind='box')
print('The median kills is {}.'.format(data.kills.median()))
print('There are {:,} observations with kills of at least one, which is {:.2%} of total observations.'.format(len([x for x in data.kills if x > 0]),len([x for x in data.kills if x > 0])/len(data.kills)))
print('killStreaks - Max number of enemy players killed in a short amount of time.')

killStreaks_describe = data.killStreaks.describe()
print(killStreaks_describe)

data.killStreaks.plot(kind='box')
print('The median killStreaks is {}.'.format(data.killStreaks.median()))
print('There are {:,} observations with killStreaks of at least one, which is {:.2%} of total observations.'.format(len([x for x in data.killStreaks if x > 0]),len([x for x in data.killStreaks if x > 0])/len(data.killStreaks)))
print('longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.')

longestKill_describe = data.longestKill.describe()
print(longestKill_describe)

data.longestKill.plot(kind='box')
print('The median longestKill is {}.'.format(data.longestKill.median()))
print('There are {:,} observations with longestKill of more than zero, which is {:.2%} of total observations.'.format(len([x for x in data.longestKill if x > 0]),len([x for x in data.longestKill if x > 0])/len(data.longestKill)))
print('The minimum longestKill is {}.'.format(min([x for x in data.longestKill if x != 0])))
print('matchDuration - Duration of match in seconds.')

matchDuration_describe = data.matchDuration.describe()
print(matchDuration_describe)

data.matchDuration.plot(kind='box')
print('The median matchDuration is {}.'.format(data.matchDuration.median()))
print('The minimum matchDuration is {} seconds.'.format(min(data.matchDuration)))
print('maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.')

maxPlace_describe = data.maxPlace.describe()
print(maxPlace_describe)

data.maxPlace.plot(kind='box')
print('The median maxPlace is {}.'.format(data.maxPlace.median()))
print('The minimum maxPlace is {}.'.format(min(data.maxPlace)))
print('numGroups - Number of groups we have data for in the match.')

numGroups_describe = data.numGroups.describe()
print(numGroups_describe)

data.numGroups.plot(kind='box')
print('The median numGroups is {}.'.format(data.numGroups.median()))
print('rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.')

rankPoints_describe = data.rankPoints.describe()
print(rankPoints_describe)

data.rankPoints.plot(kind='box')
print('The median rankPoints is {}.'.format(data.rankPoints.median()))
print('There are {:,} observations with rankPoints equals -1 or 0, which is {:.2%} of total observations.'.format(len([x for x in data.rankPoints if x <= 0]),len([x for x in data.rankPoints if x <= 0])/len(data.rankPoints)))

print('The lowest rank other than -1 and 0 is {}.'.format(min([x for x in data.rankPoints if x > 0])))
print('revives - Number of times this player revived teammates.')
print('Revives is only possible in non-solo games')
revives_describe = data.revives.describe()
print(revives_describe)

data.revives.plot(kind='box')
print('The median revives is {}.'.format(data.revives.median()))
print('There are {:,} observations with revives of at least one, which is {:.2%} of total observations.'.format(len([x for x in data.revives if x > 0]),len([x for x in data.revives if x > 0])/len(data.revives)))
print('rideDistance - Total distance traveled in vehicles measured in meters.')
print('For reference, the entire map is a 8km x 8km map.')
rideDistance_describe = data.rideDistance.describe()
print(rideDistance_describe)

data.rideDistance.plot(kind='box')
print('The median rideDistance is {}.'.format(data.rideDistance.median()))
print('There are {:,} observations with rideDistance of more than zero, which is {:.2%} of total observations.'.format(len([x for x in data.rideDistance if x > 0]),len([x for x in data.rideDistance if x > 0])/len(data.rideDistance)))
print('roadKills - Number of kills while in a vehicle.')

roadKills_describe = data.roadKills.describe()
print(roadKills_describe)

data.roadKills.plot(kind='box')
print('The median roadKills is {}.'.format(data.roadKills.median()))
print('There are {:,} observations with roadKills of at least one, which is {:.2%} of total observations.'.format(len([x for x in data.roadKills if x > 0]),len([x for x in data.roadKills if x > 0])/len(data.roadKills)))
print('swimDistance - Total distance traveled by swimming measured in meters.')
print('For reference, the entire map is a 8km x 8km map.')
swimDistance_describe = data.swimDistance.describe()
print(swimDistance_describe)

data.swimDistance.plot(kind='box')
print('The median swimDistance is {}.'.format(data.swimDistance.median()))
print('There are {:,} observations with swimDistance of more than zero, which is {:.2%} of total observations.'.format(len([x for x in data.swimDistance if x > 0]),len([x for x in data.swimDistance if x > 0])/len(data.swimDistance)))
print('teamKills - Number of times this player killed a teammate.')
print('There is friendly fire in PUBG games.')
teamKills_describe = data.teamKills.describe()
print(teamKills_describe)

data.teamKills.plot(kind='box')
print('The median teamKills is {}.'.format(data.teamKills.median()))
print('There are {:,} observations with teamKills of at least one, which is {:.2%} of total observations.'.format(len([x for x in data.teamKills if x > 0]),len([x for x in data.teamKills if x > 0])/len(data.teamKills)))
print('vehicleDestroys - Number of vehicles destroyed.')

vehicleDestroys_describe = data.vehicleDestroys.describe()
print(vehicleDestroys_describe)

data.vehicleDestroys.plot(kind='box')
print('The median vehicleDestroys is {}.'.format(data.vehicleDestroys.median()))
print('There are {:,} observations with vehicleDestroys of at least one, which is {:.2%} of total observations.'.format(len([x for x in data.vehicleDestroys if x > 0]),len([x for x in data.vehicleDestroys if x > 0])/len(data.vehicleDestroys)))
print('walkDistance - Total distance traveled on foot measured in meters.')
print('For reference, the entire map is a 8km x 8km map.')
walkDistance_describe = data.walkDistance.describe()
print(walkDistance_describe)

data.walkDistance.plot(kind='box')
print('The median walkDistance is {}.'.format(data.walkDistance.median()))
print('There are {:,} observations with walkDistance of more than zero, which is {:.2%} of total observations.'.format(len([x for x in data.walkDistance if x > 0]),len([x for x in data.walkDistance if x > 0])/len(data.walkDistance)))
print('weaponsAcquired - Number of weapons picked up.')

weaponsAcquired_describe = data.weaponsAcquired.describe()
print(weaponsAcquired_describe)

data.weaponsAcquired.plot(kind='box')
print('The median weaponsAcquired is {}.'.format(data.weaponsAcquired.median()))
print('There are {:,} observations with weaponsAcquired of at least one, which is {:.2%} of total observations.'.format(len([x for x in data.weaponsAcquired if x > 0]),len([x for x in data.weaponsAcquired if x > 0])/len(data.weaponsAcquired)))
print('winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.')

winPoints_describe = data.winPoints.describe()
print(winPoints_describe)

data.winPoints.plot(kind='hist')

print('There are {:,} observations with winPoints equals zero, which is {:.2%} of total observations.'.format(len([x for x in data.winPoints if x == 0]),len([x for x in data.winPoints if x == 0])/len(data.winPoints)))
print('The lowest rank other than zero is {}.'.format(min([x for x in data.winPoints if x != 0])))
print('winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.')

winPlacePerc_describe = data.winPlacePerc.describe()
print(winPlacePerc_describe)

data.winPlacePerc.plot(kind='hist')
print('The median winPlacePerc is {}.'.format(data.winPlacePerc.median()))
cat_unique = {}
for x in cat_var:
    cat_unique[x] = eval(x+'_describe')['unique']/eval(x+'_describe')['count'] * 100

cat_summary = pd.DataFrame([x for x in cat_unique.values()], index=cat_var, columns=['Percent Unique'])
cat_summary
cont_median = {}
for x in cont_var:
    cont_median[x] = [data.loc[:,x].median(), len([y for y in data.loc[:,x] if y > 0])/len(data.loc[:,x])*100]

cont_summary = pd.DataFrame([x for x in cont_median.values()], index=cont_var, columns=['Median','Percent Non-zero'])
cont_summary.sort_values('Percent Non-zero', ascending=False)