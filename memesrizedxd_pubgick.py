# For autoreloading modules
%load_ext autoreload
%autoreload 2
# For notebook plotting
%matplotlib inline

# Import dependencies
import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pdpbox import pdp
from plotnine import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

# Machine Learning
import sklearn
from sklearn import metrics
from scipy.cluster import hierarchy as hc
from fastai.imports import *
#from fastai.structured import *

# Import dataset
KAGGLE_DIR = '../input/'
train = pd.read_csv(KAGGLE_DIR + 'train_V2.csv')
test = pd.read_csv(KAGGLE_DIR + 'test_V2.csv')

#print(len(train))
#print(train['matchType'].unique())
#train.loc[(train['matchType'] == 'crashtpp')]
#len(train.groupby('Id'))
# Types, Data points, memory usage, etc.
#train.info()

# Check dataframe's shape
#print('\nShape of training set: ', train.shape)
# Check row with NaN value
#train[train['winPlacePerc'].isnull()]
train.drop(2744604, inplace=True)

train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
"""
plt.figure(figsize=(15,10))
sns.countplot(train[train['playersJoined']>=75]['playersJoined'])
plt.title('playersJoined')
plt.show()
"""

# Create normalized features
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['DBNOsNorm'] = train['DBNOs']*((100-train['playersJoined'])/100 + 1)
#to_show = ['Id', 'kills','killsNorm', 'DBNOs', 'DBNOsNorm','damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm', 'matchDuration', 'matchDurationNorm']
#train[to_show][0:11]

# Create new feature healsandboosts
train['healsandboosts'] = train['heals'] + train['boosts']

"""
We try to identify cheaters by checking if people are getting kills without moving.
We first identify the totalDistance travelled by a player and then set a boolean value to True
if someone got kills without moving a single inch. We will remove cheaters in our outlier detection
section.
"""
train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

# Create headshot_rate feature
train['headshot_rate'] = train['headshotKills'] / train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)

#train[train['killsWithoutMoving'] == True].head(5)
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
train.drop('killsWithoutMoving', axis=1, inplace=True)

# Players who got more than 10 roadKills
train.drop(train[train['roadKills'] > 10].index, inplace=True)

# Players who got more than 30 kills
#train[train['kills'] > 35].groupby('matchType').mean()
train.drop(train[train['kills'] > 35].index, inplace=True)

# Remove outliers
train.drop(train[(train['headshot_rate'] == 1) & (train['kills'] > 11)].index,inplace=True)

# Remove outliers
train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)

# Remove outliers
train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)

# Remove outliers
train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)

# Remove outliers
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)

# Remove outliers
train.drop(train[train['heals'] >= 40].index, inplace=True)
top = ['walkDistance','walkDistance','damageDealt']
bot = ['boosts','kills','kills']
for t,b in zip(top,bot):
    train.loc[(train[b]!=0),t+'/'+b] = train[t]/train[b]
    train.loc[(train[b]==0),t+'/'+b] = 0
print('a')
#train.head(10)
#train.sort_values('matchId').head()

#.agg([np.sum, np.mean, np.std])
#агрегации по пати, дальше ещё относительно этих данных посчитаем фичи
features_to_agg = ['groupId','DBNOs','kills','totalDistance','headshot_rate',
                   'damageDealt','killPlace','walkDistance/boosts','walkDistance/kills','damageDealt/kills']
aggs = train[features_to_agg].groupby('groupId').agg([np.sum, np.mean, np.std])

#из-за мультииндекса делаю такой костыль, если знаете как лучше через пандас, переделайте плз
train_slice = train[['groupId']]
for col in list(aggs.columns.levels[0]):
    for ag in list(aggs.columns.levels[1]):
        train_slice = train_slice.join(aggs[col,ag], on='groupId')
train_slice.columns = train_slice.columns.map(lambda x: x[0]+'_'+x[1] if type(x)==tuple else x)
train_slice.fillna(0, inplace=True)
del(aggs)
train_slice.drop(columns='groupId', inplace=True)
train = pd.concat([train, train_slice], axis=1)
#train.sort_values('groupId').head(10)
#train = train.merge(train_slice, how='inner', on='groupId')
print(len(train))
train.head()
del(train_slice)
#train.info()
#features_to_agg_match = features_to_agg+['matchId']
##features_to_agg_match.remove('groupId')
#match_mean = train[features_to_agg_match].groupby('matchId').mean().reset_index()
#print(len(match_mean))
#match_mean.head()
#full = train[['Id','matchId']].merge(match_mean, on='matchId', how='inner')
#full.columns = full.columns.map(lambda x: x+'match_mean' if (x!='Id' or x!='matchId') else x)
#full.head()
#col = list(full.columns)
#col[0] = 'Id'
#col[1] = 'matchId'
#full.columns = col
#full.head()
#len(full)
#full.drop(columns='matchId', inplace=True)
#train = train.merge(full, on='Id', how='inner')
#train.head()
#del(match_mean)
#del(full)
features = list(train.columns)
for i in col:
    features.remove(i)
features.remove('matchType')
features.remove('groupId')
features.append('matchId')
#print(features)
ranked = train.groupby('matchId')[features].rank(pct=True)
ranked.columns = ranked.columns.map(lambda x: x + '_rank')
#ranked.head()
#print(len(ranked))
train = pd.concat([train, ranked], axis=1)
del(ranked)
#train.head()
#print('There are {} different Match types in the dataset.'.format(train['matchType'].nunique()))

# One hot encode matchType
#train_slice = pd.get_dummies(train_slice, columns=['matchType'])
train.drop(columns=['matchType', 'groupId','matchId', 'Id'], inplace=True)
# Take a look at the encoding
#matchType_encoding = train_slice.filter(regex='matchType')
#matchType_encoding.head()

# Turn groupId and match Id into categorical types
#train_slice['groupId'] = train_slice['groupId'].astype('category')
#train_slice['matchId'] = train_slice['matchId'].astype('category')

# Get category coding for groupId and matchID
#train_slice['groupId_cat'] = train_slice['groupId'].cat.codes
#train_slice['matchId_cat'] = train_slice['matchId'].cat.codes

# Get rid of old columns
#train_slice.drop(columns=['groupId', 'matchId','Id'], inplace=True)

# Lets take a look at our newly created features
#train[['groupId_cat', 'matchId_cat']].head()
#train.loc[train['matchId_cat']==1, ['groupId_cat', 'matchId_cat','groupId','matchId']]
# Take sample for debugging and exploration
from sklearn.preprocessing import StandardScaler
sample = 3200000
df_sample = train.sample(sample)
scaler = StandardScaler()
# Split sample into training data and target variable
df = df_sample.drop(columns = ['winPlacePerc','winPlacePerc_rank']) #all columns except target
y = df_sample['winPlacePerc'] # Only target variable
del(df_sample)
# Function for splitting training and validation data
def split_vals(a, n : int): 
    return a[:n].copy(), a[n:].copy()
val_perc = 0.15 # % to use for validation set
n_valid = int(val_perc * sample) 
n_trn = len(df)-n_valid
# Split data
#raw_train, raw_valid = split_vals(df_sample, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
train_columns = list(X_train.columns)
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
print('Sample train shape: ', X_train.shape, 
      'Sample target shape: ', y_train.shape, 
      'Sample validation shape: ', X_valid.shape)

# Metric used for the PUBG competition (Mean Absolute Error (MAE))
from sklearn.metrics import mean_absolute_error as mae

# Function to print the MAE (Mean Absolute Error) score
# This is the metric used by Kaggle in this competition
def print_score(m : RandomForestRegressor):
    res = ['mae train: ', mae(m.predict(X_train), y_train), 
           'mae val: ', mae(m.predict(X_valid), y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
from sklearn.ensemble import RandomForestRegressor

pi_model = RandomForestRegressor(n_estimators=10, n_jobs=-1).fit(X_train, y_train)

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(pi_model, random_state=1).fit(X_valid, y_valid)
#eli5.show_weights(perm, feature_names = train_columns)
#pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', 100)  # or 1000
#pd.set_option('display.max_colwidth', -1)  # or 199
fi_pi = pd.DataFrame({'importance':perm.feature_importances_, 'column':train_columns})
to_keep = fi_pi.sort_values('importance', ascending=False)[0:80]['column']
to_keep = list(to_keep)
"""['killPlace_rank',
 'totalDistance_mean_rank',
 'killPlace_mean_rank',
 'walkDistance/kills_rank',
 'walkDistance_rank',
 'killPlace_mean',
 'kills_rank',
 'killPlace_sum',
 'killsNorm_rank',
 'kills_mean_rank',
 'totalDistance_sum_rank',
 'killStreaks_rank',
 'walkDistance',
 'totalDistance_std',
 'numGroups',
 'kills_mean',
 'killPlace_sum_rank',
 'totalDistance_mean',
 'damageDealt/kills_rank',
 'walkDistance/kills',
 'killPlace_std',
 'longestKill_rank',
 'kills_sum',
 'killPlace_std_rank',
 'kills_sum_rank',
 'kills_std',
 'totalDistance_rank',
 'boosts_rank',
 'damageDealt_sum',
 'walkDistance/boosts_mean_rank',
 'totalDistance_sum',
 'totalDistance_std_rank',
 'damageDealt_sum_rank',
 'walkDistance/kills_mean_rank',
 'damageDealt_mean',
 'kills_std_rank',
 'totalDistance',
 'weaponsAcquired_rank',
 'walkDistance/kills_mean',
 'damageDealt/kills_mean_rank',
 'walkDistance/kills_std',
 'DBNOs_mean_rank',
 'healsandboosts_rank',
 'walkDistance/kills_sum_rank',
 'boosts',
 'walkDistance/boosts_sum_rank',
 'killPlace',
 'walkDistance/kills_std_rank',
 'damageDealt/kills_mean',
 'walkDistance/boosts_mean',
 'walkDistance/kills_sum',
 'damageDealt_std',
 'damageDealt/kills_std_rank',
 'maxPlaceNorm',
 'matchDurationNorm',
 'walkDistance/boosts_std_rank',
 'damageDealt/kills_std',
 'longestKill',
 'damageDealt_mean_rank',
 'damageDealt/kills_sum_rank',
 'matchDuration',
 'damageDealt/kills_sum',
 'rideDistance_rank',
 'damageDealt_std_rank',
 'headshot_rate_rank',
 'walkDistance/boosts_sum',
 'kills',
 'walkDistance/boosts_std',
 'killsNorm',
 'DBNOs_sum_rank']
 """
len(to_keep)
val_perc = 0.15 # % to use for validation set
n_valid = int(val_perc * sample) 
n_trn = len(df)-n_valid
# Split data
#raw_train, raw_valid = split_vals(df_sample, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
X_train = X_train[to_keep]
X_valid = X_valid[to_keep]
train_columns = list(X_train.columns)
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
print('Sample train shape: ', X_train.shape, 
      'Sample target shape: ', y_train.shape, 
      'Sample validation shape: ', X_valid.shape)
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
params = {
    'boosting_type': 'goss',
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    #'bagging_fraction': 0.8,
    #'bagging_freq': 5,
    'verbose': 0,
    'n_estimators':10000,
}
params = {"objective" : "regression", "metric" : "mae", 'n_estimators':15000,
          'early_stopping_rounds':100, "num_leaves" : 40, "learning_rate" : 0.05,
          "bagging_fraction" : 0.9, "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7
         }
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                #early_stopping_rounds=1000,
                verbose_eval=1000)
display(mae(gbm.predict(X_valid, num_iteration=gbm.best_iteration),y_valid))
train_features = df.columns
#train_features = 
imp_df = pd.DataFrame()
imp_df["feature"] = list(train_features)
imp_df["importance_gain"] = gbm.feature_importance(importance_type='gain')
imp_df["importance_split"] = gbm.feature_importance(importance_type='split')
#imp_df['trn_score'] = mae(y_train, gbm.predict(X_train,num_iteration=gbm.best_iteration))
(imp_df['importance_gain']<100).index
imp_df.sort_values(by='importance_split').tail(20)
# Correlation heatmap
#corr = df_keep.corr()

# Set up the matplotlib figure
#f, ax = plt.subplots(figsize=(11, 9))

# Create heatmap
#heatmap = sns.heatmap(corr)
# Use this code if you want to save the figure
#fig = heatmap.get_figure()
#fig.savefig("Heatmap(TopFeatures).png")
def sub(test):    
    # Add engineered features to the test set

    test['headshot_rate'] = test['headshotKills'] / test['kills']
    test['headshot_rate'] = test['headshot_rate'].fillna(0)
    test['totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']
    test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')
    test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)
    test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)
    test['maxPlaceNorm'] = test['maxPlace']*((100-test['playersJoined'])/100 + 1)
    test['matchDurationNorm'] = test['matchDuration']*((100-test['playersJoined'])/100 + 1)
    test['DBNOsNorm'] = test['DBNOs']*((100-test['playersJoined'])/100 + 1)
    test['healsandboosts'] = test['heals'] + test['boosts']
    #test['killsWithoutMoving'] = ((test['kills'] > 0) & (test['totalDistance'] == 0))
    
    top = ['walkDistance','walkDistance','damageDealt']
    bot = ['boosts','kills','kills']
    for t,b in zip(top,bot):
        test.loc[(test[b]!=0),t+'/'+b] = test[t]/test[b]
        test.loc[(test[b]==0),t+'/'+b] = 0
    features_to_agg = ['groupId','DBNOs','kills','totalDistance','headshot_rate',
                       'damageDealt','killPlace','walkDistance/boosts','walkDistance/kills','damageDealt/kills']
    aggs = test[features_to_agg].groupby('groupId').agg([np.sum, np.mean, np.std])

    test_slice = test[['groupId']]
    for col in list(aggs.columns.levels[0]):
        for ag in list(aggs.columns.levels[1]):
            test_slice = test_slice.join(aggs[col,ag], on='groupId')
    test_slice.columns = test_slice.columns.map(lambda x: x[0]+'_'+x[1] if type(x)==tuple else x)
    test_slice.fillna(0, inplace=True)
    del(aggs)
    test_slice.drop(columns='groupId', inplace=True)
    test = pd.concat([test, test_slice], axis=1)
    del(test_slice)
    features = list(test.columns)
    features.remove('Id')
    features.remove('matchType')
    features.remove('groupId')
    print(features)
    ranked = test.groupby('matchId')[features].rank(pct=True)
    ranked.columns = ranked.columns.map(lambda x: x + '_rank')
    test = pd.concat([test, ranked], axis=1)
    del(ranked)
    test.drop(columns=['matchType', 'groupId','matchId', 'Id'], inplace=True)
    test = test[train_columns]
    test.fillna(0, inplace=True)
    test = scaler.fit_transform(test)
    # Remove irrelevant features from the test set
    
    # Fill NaN with 0 (temporary)
    return test
#del(train)
#del(df)
#del(X_train)
#del(X_valid)
test = pd.read_csv(KAGGLE_DIR + 'test_V2.csv')
id_ = test['Id']
test = sub(test)
np.shape(test)
# Make submission ready for Kaggle
# We use our final Random Forest model (m3) to get the predictions
predictions = np.clip(a = gbm.predict(test, num_iteration=gbm.best_iteration), a_min=0.0, a_max=1.0)
pred_df = pd.DataFrame({'Id' : id_, 'winPlacePerc' : predictions})
pred_df
# Create submission file

pred_df.to_csv("submission_2.csv", index=False)
# Last check of submission
print('Head of submission: ')
display(pred_df.head())
print('Tail of submission: ')
display(pred_df.tail())
