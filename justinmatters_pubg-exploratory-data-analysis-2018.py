# library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
# optional suppression of warning
import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')
# data import
pubg_train = pd.read_csv('../input/train_V2.csv')
pubg_test = pd.read_csv('../input/test_V2.csv')
# check the train and test data are from the same distribution
# first trin
pubg_train_stats =  pubg_train.describe()
pubg_train_stats


# then test
pubg_test_stats =  pubg_test.describe()
pubg_test_stats
# and finally subtract one from the other and crosscheck magnitures
pubg_train_stats.drop(columns = 'winPlacePerc')
train_test_difference = pubg_train_stats - pubg_test_stats
train_test_difference
# lets start with a basic correlation plot
corr_vals = pubg_train.corr()

fig, axes = plt.subplots(figsize=(15,15))
sns.heatmap(corr_vals, ax = axes, cmap="rainbow", annot=True);
# lets see how each individual variable is related to the target variable
# note that to finish in a reasonable time we need to take a subset
# of the training data
pubg_small = pubg_train[0:-1:100]

# select interesting correlated columns
column_list = [
    'damageDealt', 'DBNOs', 'heals',
    'killPlace',  'kills', 'boosts',
    'killStreaks',  'rideDistance', 'walkDistance',
    'weaponsAcquired']
pubg_clipped = pubg_small[column_list+['winPlacePerc']]

# clip off extremes to get nice plots
pubg_clipped = pubg_clipped.clip(
    lower=None, upper= pubg_clipped.quantile(0.999), 
    axis = 1)

#ycle though columns to look at correlation
for column in column_list:
    #print(column)
    sns.jointplot(x = column, y = "winPlacePerc", data = pubg_clipped, kind = "kde")
#cycle though columns to look at how linear the correlation is
for column in column_list:
    #print(column)
    fig, axes = plt.subplots(figsize=(6,6))
    ax = sns.regplot(x = column, y = "winPlacePerc", data = pubg_clipped,
        scatter_kws={"s": 80}, order=3, line_kws={'color':'red'},
        robust = False, ci=None, truncate=True)
# set up our feature engineering labels

# list of the variables suspected to be significant in data analysis
variables = ['killPlace', 'boosts', 'walkDistance', 'weaponsAcquired', 'damageDealt', 'heals', 
             'kills', 'longestKill', 'killStreaks', 'rideDistance','rampage', 'lethality', 
             'items', 'totalDistance']

keep_labels = variables + ['matchId','groupId', 'matchType', 'winPlacePerc']

def feature_engineering(pubg_data):
    '''FEATURE ENGINEERING
    GIVEN: a PUBG dataframe which must have a dummy 'winPlacePerc' column if a test set
    Conduct data engineering including:
    producing group data, normalising data with relevant match stats, clipping extreme results
    RETURNS: pubg_x dataframe consisting of feature engineered input columns
             pubg_y dataframe with target values (0 dummy frame if this is a test set)
    '''

    # total the pickups
    pubg_data['items'] = pubg_data['heals'] + pubg_data['boosts'] + pubg_data["weaponsAcquired"]
    
    # total the distance
    pubg_data['totalDistance'] = pubg_data['rideDistance'] + pubg_data['swimDistance'] + pubg_data['walkDistance']

    # estimate accuracy of players
    pubg_data['lethality'] = pubg_data['headshotKills'] / pubg_data['kills']
    pubg_data['lethality'].replace(np.inf, 0, inplace=True)
    pubg_data['lethality'].fillna(0, inplace=True)
    
    # estimate how players behave in shootouts
    pubg_data['rampage'] = pubg_data['killStreaks'] / pubg_data['kills']
    pubg_data['rampage'].replace(np.inf, 0, inplace=True)
    pubg_data['rampage'].fillna(0, inplace=True)
    
    # reduce dataframe to the columns we want to use
    pubg_data = pubg_data[keep_labels]

    # use groupby to get means for team
    pubg_group_means = pubg_data.groupby(['matchId','groupId']).mean().reset_index()

    # use groupby to get means of matches
    pubg_match_means = pubg_data.groupby(['matchId']).mean().reset_index()

    # merge back in leaving columns unchanged for one set to allow for future suffixing (only affects shared columns)
    pubg_engineered = pd.merge(pubg_data, pubg_group_means, 
                               suffixes=["", "_group"], how = "left", on = ['matchId', 'groupId']) 
    pubg_engineered = pd.merge(pubg_engineered, pubg_match_means, 
                               suffixes=["_player", "_match"], how = "left", on = ['matchId'])

    # norm the player variables
    for variable in variables:
        pubg_engineered[variable+'_norm'] = pubg_engineered[variable+'_player']/(pubg_engineered[variable+'_match']+0.1)

    # norm the group variables
    for variable in variables:
        pubg_engineered[variable+'_g_norm'] = pubg_engineered[variable+'_group']/(pubg_engineered[variable+'_match']+0.1)
        
    # one hot encode the matchTypes since different matches may follow different logics
    one_hot = pd.get_dummies(pubg_engineered['matchType'])
    # Drop column B as it is now encoded
    pubg_engineered = pubg_engineered.drop('matchType',axis = 1)
    # Join the encoded df
    pubg_engineered = pubg_engineered.join(one_hot)
    pubg_engineered.drop(columns = ['winPlacePerc_group', 'winPlacePerc_match'], inplace = True)
    pubg_engineered.rename(columns = {'winPlacePerc_player': 'winPlacePerc'}, inplace = True)
    pubg_engineered = pubg_engineered.reset_index(drop=True)
    
    return pubg_engineered
# must engineer on the full dataset to get correct group and match means
pubg_engineered = feature_engineering(pubg_train)
# group related cloumns together
pubg_engineered = pubg_engineered.sort_index(axis=1)
# now we can compare the correlation of different 
# versions of the variable quite easily

# grab our columns
available_columns = list(pubg_engineered.columns.values)
# work out where each group of variables we want to compare start
start_values = [0,7,17,22,27,32,37,42,47,59,64,73,78,83] 
# loop over our subsets creating correlation plots
for start in start_values:
    column_selection =  available_columns[
        start: start+5] + ['winPlacePerc']
    corr_vals = pubg_engineered[column_selection].corr()

    fig, axes = plt.subplots(figsize=(5,5))
    sns.heatmap(corr_vals, ax = axes, 
                cmap="rainbow", annot=True);