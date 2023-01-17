import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.head()
# Remove the row with the missing target value
train = train[train['winPlacePerc'].isna() != True]

# Add a feature containing the number of players that joined each match.
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['headshotKillsNorm'] = train['headshotKills']*((100-train['playersJoined'])/100 + 1)
train['killPlaceNorm'] = train['killPlace']*((100-train['playersJoined'])/100 + 1)
train['killPointsNorm'] = train['killPoints']*((100-train['playersJoined'])/100 + 1)
train['killStreaksNorm'] = train['killStreaks']*((100-train['playersJoined'])/100 + 1)
train['longestKillNorm'] = train['longestKill']*((100-train['playersJoined'])/100 + 1)
train['roadKillsNorm'] = train['roadKills']*((100-train['playersJoined'])/100 + 1)
train['teamKillsNorm'] = train['teamKills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['DBNOsNorm'] = train['DBNOs']*((100-train['playersJoined'])/100 + 1)
train['revivesNorm'] = train['revives']*((100-train['playersJoined'])/100 + 1)

    
# Features to remove
train = train.drop([ 'kills', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks', 
 'longestKill', 'roadKills', 'teamKills', 'damageDealt', 'DBNOs', 'revives'],axis=1)
train.head()
# Total distance travelled
train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']

# Normalise the matchTypes to standard fromat
def standardize_matchType(data):
    data['matchType'][data['matchType'] == 'normal-solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'normal-duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'flaretpp'] = 'Other'
    data['matchType'][data['matchType'] == 'flarefpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashtpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashfpp'] = 'Other'

    return data


data = standardize_matchType(train)
#print (set(data['matchType']))
# We can do a sanity check of the data, making sure we have the new 
# features created and the matchType feature is standardised.
data.head()
# Seperate the data into the matchTypes
solo = data[data['matchType'] == 'Solo']
duo = data[data['matchType'] == 'Duo']
squad = data[data['matchType'] == 'Squad']
other = data[data['matchType'] == 'Other']
# SOLO: Features to keep
solo_features = ['boosts','heals', 'rideDistance','walkDistance','weaponsAcquired',
                 # Engineered Features
                 'damageDealtNorm','headshotKillsNorm','killPlaceNorm',
                 'killsNorm','killStreaksNorm','longestKillNorm',
                 'playersJoined','totalDistance']

solo = solo[solo_features]
solo.head()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(solo.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
# DUO: Features to keep
duo_features = ['assists','boosts', 'heals','rideDistance','walkDistance',
                'weaponsAcquired',
                # Engineered Features
                'damageDealtNorm','DBNOsNorm', 'killPlaceNorm',
                'killsNorm','killStreaksNorm','longestKillNorm',
                'revivesNorm', 'playersJoined','totalDistance']

duo = duo[duo_features]
duo.head()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(duo.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
# SQUAD: Features to keep
squad_features = ['assists','boosts','heals','rideDistance',
                  'walkDistance','weaponsAcquired',
                  # Engineered Features
                  'damageDealtNorm','DBNOsNorm', 'killPlaceNorm',
                  'killsNorm','killStreaksNorm','longestKillNorm',
                  'revivesNorm','playersJoined','totalDistance']

squad = squad[squad_features]
squad.head()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(squad.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
# OTHER: Features to keep
other_features = ['assists','boosts','heals','rideDistance',
                  'walkDistance','weaponsAcquired',
                  # Engineered Features
                  'damageDealtNorm','DBNOsNorm','headshotKillsNorm',
                  'killPlaceNorm','killsNorm','killStreaksNorm','longestKillNorm',
                  'revivesNorm','playersJoined','totalDistance']

other = other[other_features]
other.head()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(other.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()



