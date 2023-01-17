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
train = train[0:1000000]
train.head()
test.head()
# Remove the row with the missing target value
train = train[train['winPlacePerc'].isna() != True]

# Add a feature containing the number of players that joined each match.
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

# Lets look at only those matches with more than 50 players.
#data = train[train['playersJoined'] > 50]

#plt.figure(figsize=(15,15))
#sns.countplot(data['playersJoined'].sort_values())
#plt.title('Number of players joined',fontsize=15)
#plt.show()
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
    data['matchType'][data['matchType'] == 'solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'normal-duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo'] = 'Duo'
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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

# Create a master copy of the data, so we can restore the default features.
#train_master = train_scaled.copy()

# Extract the target variable.
y = train_scaled['winPlacePerc']
X = train_scaled.drop(['winPlacePerc'],axis=1)

# Split the data in to training and validation set
size = 0.3
seed = 42
   
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=size, random_state=seed)

# The function takes the training and validation data to fit and score a group of models
def runAllModels(X_train, X_validation, Y_train, Y_validation):
        
    linear = LinearRegression(copy_X=True)
    linear.fit(X_train,Y_train)
    print("Linear Model score: {0:.3f}%".format(linear.score(X_validation,Y_validation)*100))

    ridge = Ridge(copy_X=True)
    ridge.fit(X_train,Y_train)
    print("Ridge Model score: {0:.3f}%".format(ridge.score(X_validation,Y_validation)*100))

    lasso = Lasso(copy_X=True)
    lasso.fit(X_train,Y_train)
    print("Lasso Model score: {0:.3f}%".format(lasso.score(X_validation,Y_validation)*100))

    elastic = ElasticNet(copy_X=True)
    elastic.fit(X_train,Y_train)
    print("ElasticNet Model score: {0:.3f}%".format(elastic.score(X_validation,Y_validation)*100))

    ada = AdaBoostRegressor(learning_rate=0.8)
    ada.fit(X_train,Y_train)
    print("AdaBoostRegressor Model score: {0:.3f}%".format(ada.score(X_validation,Y_validation)*100))

    GBR = GradientBoostingRegressor(learning_rate=0.8)
    GBR.fit(X_train,Y_train)
    print("GradientBoostingRegressor Model score: {0:.3f}%".format(GBR.score(X_validation,Y_validation)*100))

    forest = RandomForestRegressor(n_estimators=10)
    forest.fit(X_train,Y_train)
    print("RandomForestRegressor Model score: {0:.3f}%".format(forest.score(X_validation,Y_validation)*100))

    tree = DecisionTreeRegressor()
    tree.fit(X_train,Y_train)
    print("DecisionTreeRegressor Model score: {0:.3f}%".format(tree.score(X_validation,Y_validation)*100))

train_scaled.head()
test_scaled.head()


runAllModels(X_train, X_validation, Y_train, Y_validation)
GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(X,y)

predictions_all = GBR.predict(test_scaled)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def display_VIF(data):
    x_features=list(data)
    data_mat = data[x_features].as_matrix()                                                                                                              
    vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
    vif_factors = pd.DataFrame()
    vif_factors['Feature'] = list(x_features)
    vif_factors['VIF'] = vif
    
    return vif_factors

vif = display_VIF(train_scaled)
vif.sort_values(by=['VIF'],ascending=False)
# Drop the features with the largest VIF and check for multicollinearity
train_scaled = train_scaled.drop(['totalDistance','rideDistance','swimDistance','walkDistance','numGroups','maxPlace',
                                 'playersJoined','winPoints','rankPoints'], axis=1)
test_scaled = test_scaled.drop(['totalDistance','rideDistance','swimDistance','walkDistance','numGroups','maxPlace',
                                 'playersJoined','winPoints','rankPoints'], axis=1)

vif = display_VIF(train_scaled)
vif.sort_values(by=['VIF'],ascending=False)
# Drop the the remaining features that have a VIF greater than 10.
train_scaled = train_scaled.drop(['matchDuration','killsNorm','killPlaceNorm'], axis=1)
test_scaled = test_scaled.drop(['matchDuration','killsNorm','killPlaceNorm'], axis=1)

vif = display_VIF(train_scaled)
vif.sort_values(by=['VIF'],ascending=False)
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(train_scaled.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()

train_scaled.head()

# Train Test Split
y = train_scaled['winPlacePerc']
X = train_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=size, random_state=seed)

# Run all models with the reduced set of features.
runAllModels(X_train, X_validation, Y_train, Y_validation)

train_copy.head()
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

runAllModels(X_train, X_validation, Y_train, Y_validation)
GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(solo_X,solo_y)
print("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(solo_X,solo_y)*100))

predictions_solo = GBR.predict(solo_test_scaled)

duo_y = duo_scaled['winPlacePerc']
duo_X = duo_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(duo_X, duo_y, test_size=size, random_state=seed)

runAllModels(X_train, X_validation, Y_train, Y_validation)
GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(duo_X,duo_y)
print("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(duo_X,duo_y)*100))

predictions_duo = GBR.predict(duo_test_scaled) 
squad_y = squad_scaled['winPlacePerc']
squad_X = squad_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(squad_X, squad_y, test_size=size, random_state=seed)

runAllModels(X_train, X_validation, Y_train, Y_validation)
GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(squad_X,squad_y)
print("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(squad_X,squad_y)*100))

predictions_squad = GBR.predict(squad_test_scaled)
other_y = other_scaled['winPlacePerc']
other_X = other_scaled.drop(['winPlacePerc'],axis=1)

# We will maintain the existing size and random state parameters for repeatability.
X_train, X_validation, Y_train, Y_validation = train_test_split(other_X, other_y, test_size=size, random_state=seed)

runAllModels(X_train, X_validation, Y_train, Y_validation)
 
GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(other_X,other_y)
print("GradientBoost Model traininig: {0:.3f}%".format(GBR.score(other_X,other_y)*100))

predictions_other = GBR.predict(other_test_scaled) 
def create_submission(submission_Id, predictions, filename):
    submission = pd.DataFrame({'Id': submission_Id, 'winPlacePerc': predictions})
    
    submission.to_csv(filename+'.csv',index=False)

test_submission_solo = test_submission[test_submission['matchType'] == 'Solo']
test_submission_duo = test_submission[test_submission['matchType'] == 'Duo']
test_submission_squad = test_submission[test_submission['matchType'] == 'Squad']
test_submission_other = test_submission[test_submission['matchType'] == 'Other']

matchTypeId = test_submission_solo['Id'].append(test_submission_duo['Id']).append(test_submission_squad['Id']).append(test_submission_other['Id'])

predictions_solo[predictions_solo > 1] = 1
predictions_solo[predictions_solo < 0] = 0

predictions_duo[predictions_duo > 1] = 1
predictions_duo[predictions_duo < 0] = 0

predictions_squad[predictions_squad > 1] = 1
predictions_squad[predictions_squad < 0] = 0

predictions_other[predictions_other > 1] = 1
predictions_other[predictions_other < 0] = 0


predications_matchtype = np.append(np.append(predictions_solo,predictions_duo),np.append(predictions_squad,predictions_other))

create_submission(matchTypeId, predications_matchtype, 'submission_matchType')
