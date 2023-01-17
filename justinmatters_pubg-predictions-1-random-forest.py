# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# machine learning imports
import sklearn as skl
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor # does not auto import
from sklearn.metrics import mean_absolute_error # does not auto import
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# set up our feature engineering labels
e_labels = ['matchId','groupId','killPlace', 'boosts', 'walkDistance', 'weaponsAcquired', 'damageDealt', 'heals', 
                        'kills', 'longestKill', 'killStreaks', 'rideDistance', 'winPlacePerc', 'matchType']

# list of the variables discovered to be significant in data analysis
variables = ['killPlace', 'boosts', 'walkDistance', 'weaponsAcquired', 'damageDealt', 'heals', 
                        'kills', 'longestKill', 'killStreaks', 'rideDistance', 'winPlacePerc']

# labels for desired columns for dataset to feed to model
# use normed values for stats except for rideDistance where correlation worsens
# use category variables for game types where this information assists the model
labels = [ 
       'killPlace_g_norm', 'boosts_g_norm', 'walkDistance_g_norm',
       'weaponsAcquired_g_norm', 'damageDealt_g_norm', 'heals_g_norm',
       'kills_g_norm', 'longestKill_g_norm', 'killStreaks_g_norm',
       'rideDistance_group',  'duo', 'duo-fpp', 'solo', 'solo-fpp', 'squad',
       'squad-fpp']

def feature_engineering(pubg_data):
    '''FEATURE ENGINEERING
    GIVEN: a PUBG dataframe which must have a dummy 'winPlacePerc' column if a test set
    Conduct data engineering including:
    producing group data, normalising data with relevant match stats, clipping extreme results
    RETURNS: pubg_x dataframe consisting of feature engineered input columns
             pubg_y dataframe with target values (0 dummy frame if this is a test set)
    '''

    # reduce dataframe to the columns we want to use
    pubg_data = pubg_data[e_labels]

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

    # setting up our basic data
    pubg_data = pubg_engineered.reset_index(drop=True)

    # create raw input  data
    pubg_x = pubg_data[labels]

    # clip outliers on a per column basis 
    pubg_x = pubg_x.clip(lower=None, upper= pubg_x.quantile(0.99), axis = 1)

    # set up our target data (not needed for test, so creates a dummy variable
    pubg_y = pubg_data['winPlacePerc_player']

    #return values
    return pubg_x, pubg_y
# import our training data
pubg_data_all = pd.read_csv("../input/train_V2.csv")
pubg_data = pubg_data_all.dropna() # there is an na value we need to drop
# do our feature engineering and split off our target variable
pubg_x, pubg_y = feature_engineering(pubg_data)
# now lets scale data to ensure column scales do not skew results
scaler = skl.preprocessing.StandardScaler().fit(pubg_x)

# lets look at the head again - we need to convert back to dataframe from numpy array though
pubg_x = pd.DataFrame(scaler.transform(pubg_x), columns= labels)
# having a scaler object will let us use it on the test data too :-)
# now lets create the model
model_rf = RandomForestRegressor(n_estimators=32, oob_score=False, random_state=0, n_jobs = -1, verbose = 2)

# and fit it...
model_rf.fit(pubg_x, pubg_y)
# save the model out (not needed for kernel run)
#joblib.dump(model_rf, 'pubg_model_rf.joblib') 

# now lets test how well it fits training data (with normalisation AND extreme value clipping)
predict_train_rf = model_rf.predict(pubg_x)
print('Mean absolute error for the training set using random forest regressor model %.4f' %
      mean_absolute_error(pubg_y, np.clip(predict_train_rf, 0, 1)))
# save memory before running training data
del(pubg_data)
del(pubg_data_all)
del(pubg_x)
del(pubg_y)
del(predict_train_rf)
# now we are ready to read in the test data
pubg_data_test = pd.read_csv('../input/test_V2.csv')
#print(pubg_data_test.isnull().sum()) # no NaNs

# add a dummy winPlacePerc column to pubg_data_test so we can use our feature engineering function
pubg_data_test['winPlacePerc'] = 0
# do our feature engineering (NB pubg_y is a dummy return here)
pubg_x, pubg_y = feature_engineering(pubg_data_test)
#use our scaler on the test data too
pubg_x = pd.DataFrame(scaler.transform(pubg_x), columns= labels)

# then make predictions
predict_test_rf = model_rf.predict(pubg_x)
# and clip outlying values as they cannot be correct (NB we can likely be more sophisticated than this)
predict_test_rf_clip = np.clip(predict_test_rf, 0, 1)
# prepare output
predict_test_rf_df = pd.DataFrame(data= predict_test_rf_clip, columns=['winPlacePerc'])
output_df = pd.merge(pubg_data_test["Id"].to_frame(),predict_test_rf_df['winPlacePerc'].to_frame(), left_index=True, right_index=True)
output_df.head()

# write output
output_df.to_csv("submission2.csv", index = False, index_label=False)