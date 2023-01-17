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

train.head()
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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor

# Train Test Split
y = train_scaled['winPlacePerc']
X = train_scaled.drop(['winPlacePerc'],axis=1)
size = 0.30
seed = 42

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=size, random_state=seed)

# The function takes the training and validation data to fit and score the model
def runAllModels(X_train, X_validation, Y_train, Y_validation):
    
    
    linear = LinearRegression(copy_X=True)
    linear.fit(X_train,Y_train)
    #print("Linear Model traininig: {0:.5}%".format(linear.score(X_train,Y_train)*100))
    print("Linear Model score: {0:.5}%".format(linear.score(X_validation,Y_validation)*100))

    ridge = Ridge(copy_X=True)
    ridge.fit(X_train,Y_train)
    #print("Ridge Model traininig: {0:.5}%".format(ridge.score(X_train,Y_train)*100))
    print("Ridge Model score: {0:.5}%".format(ridge.score(X_validation,Y_validation)*100))

    lasso = Lasso(copy_X=True)
    lasso.fit(X_train,Y_train)
    #print("Lasso Model traininig: {0:.5}%".format(lasso.score(X_train,Y_train)*100))
    print("Lasso Model score: {0:.5}%".format(lasso.score(X_validation,Y_validation)*100))

    elastic = ElasticNet(copy_X=True)
    elastic.fit(X_train,Y_train)
    #print("Elastic Model traininig: {0:.5}%".format(elastic.score(X_train,Y_train)*100))
    print("Elastic Model score: {0:.5}%".format(elastic.score(X_validation,Y_validation)*100))

    ada = AdaBoostRegressor(learning_rate=0.8)
    ada.fit(X_train,Y_train)
    #print("AdaBoost Model traininig: {0:.5}%".format(ada.score(X_train,Y_train)*100))
    print("AdaBoost Model score: {0:.5}%".format(ada.score(X_validation,Y_validation)*100))

    GBR = GradientBoostingRegressor(learning_rate=0.8)
    GBR.fit(X_train,Y_train)
    #print("GradientBoost Model traininig: {0:.5}%".format(GBR.score(X_train,Y_train)*100))
    print("GradientBoost Model score: {0:.5}%".format(GBR.score(X_validation,Y_validation)*100))

    forest = RandomForestRegressor(n_estimators=10)
    forest.fit(X_train,Y_train)
    #print("RandomForest Model traininig: {0:.5}%".format(forest.score(X_train,Y_train)*100))
    print("RandomForest Model score: {0:.5}%".format(forest.score(X_validation,Y_validation)*100))

    tree = DecisionTreeRegressor()
    tree.fit(X_train,Y_train)
    #print("DecisionTree Model traininig: {0:.5}%".format(tree.score(X_train,Y_train)*100))
    print("DecisionTree Model score: {0:.5}%".format(tree.score(X_validation,Y_validation)*100))
runAllModels(X_train, X_validation, Y_train, Y_validation)
GBR = GradientBoostingRegressor(learning_rate=0.8)
GBR.fit(X,y)
#print("GradientBoost Model traininig: {0:.5}%".format(GBR.score(X,y)*100))

predictions = GBR.predict(test)
predictions[predictions > 1] = 1
predictions[predictions < 0] = 0
def create_submission(submission_Id, predictions, filename):
    submission = pd.DataFrame({'Id': submission_Id, 'winPlacePerc': predictions})
    submission.to_csv(filename+'.csv',index=False)
create_submission(test_ids, predictions, 'submission_full')
# Fit all the data to the model and return the predictions for the testing set.
def make_predictions(model, X, y, test):
    model.fit(X,y)
    #print("Model traininig: {0:.5}%".format(model.score(X,y)*100))

    return model.predict(test)
    
# Create the models to use for our predictions
linear = LinearRegression(copy_X=True)
ridge = Ridge(copy_X=True)
GBR = GradientBoostingRegressor(learning_rate=0.8)
forest = RandomForestRegressor(n_estimators=10)
tree = DecisionTreeRegressor()
    
# Get some predictions from each model
predictions_linear = make_predictions(linear, X, y, test_scaled)
predictions_ridge = make_predictions(ridge, X, y, test_scaled)
predictions_GBR = make_predictions(GBR, X, y, test_scaled)
predictions_forest = make_predictions(forest, X, y, test_scaled)
predictions_tree = make_predictions(tree, X, y, test_scaled)

# Adjust the predictions that are outside the bounds of the target variable.
predictions_linear[predictions_linear > 1] = 1
predictions_linear[predictions_linear < 0] = 0

predictions_ridge[predictions_ridge > 1] = 1
predictions_ridge[predictions_ridge < 0] = 0

predictions_GBR[predictions_GBR > 1] = 1
predictions_GBR[predictions_GBR < 0] = 0

predictions_forest[predictions_forest > 1] = 1
predictions_forest[predictions_forest < 0] = 0

predictions_tree[predictions_tree > 1] = 1
predictions_tree[predictions_tree < 0] = 0

# We create the submission files for each model
create_submission(test_ids, predictions_linear, 'submission_linear')
create_submission(test_ids, predictions_ridge, 'submission_ridge')
create_submission(test_ids, predictions_GBR, 'submission_GBR')
create_submission(test_ids, predictions_forest, 'submission_forest')
create_submission(test_ids, predictions_tree, 'submission_tree')


