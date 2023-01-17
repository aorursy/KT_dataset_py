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
train = train[0:2000000]
train.head()
test.head()
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

## Blatently stolen from Hyun Woo Kim https://www.kaggle.com/chocozzz/lightgbm-baseline

def createFeatureRates(data):
    data['headshotrate'] = data['killsNorm']/data['headshotKillsNorm']
    data['killStreakrate'] = data['killStreaksNorm']/data['killsNorm']
    data['healthitems'] = data['heals'] + data['boosts']
    data['killPlace_over_maxPlace'] = data['killPlaceNorm'] / data['maxPlace']
    data['headshotKills_over_kills'] = data['headshotKillsNorm'] / data['killsNorm']
    data['walkDistance_over_weapons'] = data['walkDistance'] / data['weaponsAcquired']
    data['walkDistance_over_heals'] = data['walkDistance'] / data['heals']
    data['walkDistance_over_kills'] = data['walkDistance'] / data['killsNorm']
    data['distance_over_weapons'] = data['totalDistance'] / data['weaponsAcquired']
    data['distance_over_heals'] = data['totalDistance'] / data['heals']
    data['distance_over_kills'] = data['totalDistance'] / data['killsNorm']
    data['killsPerDistance'] = data['killsNorm'] / data['totalDistance']
    data['skill'] = data['headshotKillsNorm'] + data['roadKillsNorm']
    
    data[data == np.Inf] = np.NaN
    data[data == np.NINF] = np.NaN
    
    data.fillna(0, inplace=True)
    
    return data
train = createFeatureRates(train)
test = createFeatureRates(test)
train.head()
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


train = standardize_matchType(train)
test = standardize_matchType(test)
# We need a copy of the test data with the player id later on
test_submission = test.copy()

train = train.drop(['Id','groupId','matchId'], axis=1)
test = test.drop(['Id','groupId','matchId'], axis=1)
# We need to keep a copy of the test data for the test id's later 
train_copy = train.copy()
test_copy = test.copy()
# Transform the matchType into scalar values
le = LabelEncoder()
train['matchType']=le.fit_transform(train['matchType'])
test['matchType']=le.fit_transform(test['matchType'])

# We can do a sanity check of the data, making sure we have the new 
# features created and the matchType feature is standardised.
train.head()
train.describe()
scaler = MinMaxScaler()

train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
test_scaled = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)

train_scaled.head()
train_scaled.describe()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Extract the target variable.
y = train_scaled['winPlacePerc']
X = train_scaled.drop(['winPlacePerc'],axis=1)

# Split the data in to training and validation set
size = 0.3
seed = 42
   
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=size, random_state=seed)

# Create a data set for each matchType and drop that feature, as there will be no variance, and hence no predictive power.
solo = train_copy[train_copy['matchType'] == 'Solo']
solo = solo.drop(['matchType'], axis=1)
duo = train_copy[train_copy['matchType'] == 'Duo']
duo = duo.drop(['matchType'], axis=1)
squad = train_copy[train_copy['matchType'] == 'Squad']
squad = squad.drop(['matchType'], axis=1)
other = train_copy[train_copy['matchType'] == 'Other']
other = other.drop(['matchType'], axis=1)

# since we used a copy of the trained data that hasn't been scaled, we need to scale the features again.
scaler = MinMaxScaler()
solo_scaled = pd.DataFrame(scaler.fit_transform(solo), columns=solo.columns)
duo_scaled = pd.DataFrame(scaler.fit_transform(duo), columns=duo.columns)
squad_scaled = pd.DataFrame(scaler.fit_transform(squad), columns=squad.columns)
other_scaled = pd.DataFrame(scaler.fit_transform(other), columns=other.columns)

# Seperate the matchType data
test_solo = test_copy[test_copy['matchType'] == 'Solo']
test_solo = test_solo.drop(['matchType'], axis=1)
test_duo = test_copy[test_copy['matchType'] == 'Duo']
test_duo = test_duo.drop(['matchType'], axis=1)
test_squad = test_copy[test_copy['matchType'] == 'Squad']
test_squad = test_squad.drop(['matchType'], axis=1)
test_other = test_copy[test_copy['matchType'] == 'Other']
test_other = test_other.drop(['matchType'], axis=1)

solo_test_scaled = pd.DataFrame(scaler.fit_transform(test_solo), columns=test_solo.columns)
duo_test_scaled = pd.DataFrame(scaler.fit_transform(test_duo), columns=test_duo.columns)
squad_test_scaled = pd.DataFrame(scaler.fit_transform(test_squad), columns=test_squad.columns)
other_test_scaled = pd.DataFrame(scaler.fit_transform(test_other), columns=test_other.columns)

solo_y = solo_scaled['winPlacePerc']
solo_X = solo_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(solo_X, solo_y, test_size=size, random_state=seed)


GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(X_train,Y_train)
print("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(X_validation,Y_validation)*100))


duo_y = duo_scaled['winPlacePerc']
duo_X = duo_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(duo_X, duo_y, test_size=size, random_state=seed)


GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(X_train,Y_train)
print("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(X_validation,Y_validation)*100))

predictions_duo = GBR.predict(duo_test_scaled) 
squad_y = squad_scaled['winPlacePerc']
squad_X = squad_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(squad_X, squad_y, test_size=size, random_state=seed)

GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(X_train,Y_train)
print("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(X_validation,Y_validation)*100))


other_y = other_scaled['winPlacePerc']
other_X = other_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(other_X, other_y, test_size=size, random_state=seed)

 
GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(X_train,Y_train)
print("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(X_validation,Y_validation)*100))

predictions_other = GBR.predict(other_test_scaled) 
GBR_solo = GradientBoostingRegressor(learning_rate=0.8)
GBR_solo.fit(solo_X,solo_y)
GBR_duo = GradientBoostingRegressor(learning_rate=0.8)
GBR_duo.fit(duo_X,duo_y)
GBR_squad = GradientBoostingRegressor(learning_rate=0.8)
GBR_squad.fit(squad_X,squad_y)
GBR_other = GradientBoostingRegressor(learning_rate=0.8)
GBR_other.fit(other_X,other_y)
predictions_solo = GBR_solo.predict(solo_test_scaled)
predictions_duo = GBR_duo.predict(duo_test_scaled)
predictions_squad = GBR_squad.predict(squad_test_scaled)
predictions_other = GBR_other.predict(other_test_scaled)

test_submission_solo = test_submission[test_submission['matchType'] == 'Solo']
test_submission_duo = test_submission[test_submission['matchType'] == 'Duo']
test_submission_squad = test_submission[test_submission['matchType'] == 'Squad']
test_submission_other = test_submission[test_submission['matchType'] == 'Other']

# Aggregate the Id's together again
matchTypeId = test_submission_solo['Id'].append(test_submission_duo['Id']).append(test_submission_squad['Id']).append(test_submission_other['Id'])

predictions_solo[predictions_solo > 1] = 1
predictions_solo[predictions_solo < 0] = 0

predictions_duo[predictions_duo > 1] = 1
predictions_duo[predictions_duo < 0] = 0

predictions_squad[predictions_squad > 1] = 1
predictions_squad[predictions_squad < 0] = 0

predictions_other[predictions_other > 1] = 1
predictions_other[predictions_other < 0] = 0

# Aggregate the final predictions from each model. This must be the same order as the aggregation of the Id's
predictions_matchtype = np.append(np.append(predictions_solo,predictions_duo),np.append(predictions_squad,predictions_other))

submission = pd.DataFrame({'Id': matchTypeId, 'winPlacePerc': predictions_matchtype})
submission.to_csv('submission_matchType.csv',index=False)
matchTypeId.size
predictions_matchtype.size
#1934174 




