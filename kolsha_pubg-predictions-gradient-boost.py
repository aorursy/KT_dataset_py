import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')

train.head(20)
# Remove the row with the missing target value
train = train[train['winPlacePerc'].isna() != True]

# Add a feature containing the number of players that joined each match.
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

# Lets look at only those matches with more than 50 players.
data = train[train['playersJoined'] > 50]

plt.figure(figsize=(15,15))
sns.countplot(data['playersJoined'].sort_values())
plt.title('Number of players joined',fontsize=15)
plt.show()
def normaliseFeatures(train):
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

    # Remove the original features we normalised
    train = train.drop(['kills', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks', 
                        'longestKill', 'roadKills', 'teamKills', 'damageDealt', 'DBNOs', 'revives'],axis=1)

    return train

train = normaliseFeatures(train)
test = normaliseFeatures(test)
train.head()
# Total distance travelled
train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']
test['totalDistance'] = test['walkDistance'] + test['rideDistance'] + test['swimDistance']

# Normalise the matchTypes to standard fromat
def standardize_matchType(data):
    data['matchType'][data['matchType'] == 'normal-solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'normal-duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'squad'] = 'Squad'
    data['matchType'][data['matchType'] == 'squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'flaretpp'] = 'Other'
    data['matchType'][data['matchType'] == 'flarefpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashtpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashfpp'] = 'Other'

    return data


train = standardize_matchType(train)
test = standardize_matchType(test)
train = train.drop(['Id','groupId','matchId'], axis=1)
# Save the Ids for the submission later on
test_ids = test['Id']
test = test.drop(['Id','groupId','matchId'], axis=1)
# Transform the matchType into scalar values
le = LabelEncoder()
train['matchType']=le.fit_transform(train['matchType'])
test['matchType']=le.fit_transform(test['matchType'])
# We can do a sanity check of the data, making sure we have the new 
# features created and the matchType feature is standardised.
train.head()
test.head()
train.describe()
scaler = MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
test_scaled = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)

train_scaled.head()
train_scaled.describe()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Train Test Split
y = train_scaled['winPlacePerc']
X = train_scaled.drop(['winPlacePerc'],axis=1)
size = 0.30
seed = 42

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=size, random_state=seed)

GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(X,y)

predictions = GBR.predict(test)
predictions[predictions > 1] = 1
predictions[predictions < 0] = 0
submission = pd.DataFrame({'Id': test_ids, 'winPlacePerc': predictions})
submission.to_csv('submission_GBR.csv',index=False)
