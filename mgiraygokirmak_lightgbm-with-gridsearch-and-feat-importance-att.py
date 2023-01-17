# Import necessary everyday os libs
import gc
import sys

# Import the usual suspects
import numpy as np
import pandas as pd
# Universal pandas dataframe memory footprint reducer for those dealing with big data but not that big that require spark
def df_footprint_reduce(df, skip_obj=False, skip_int=False, skip_float=False, print_comparison=True):
    '''
    :param df              : Pandas Dataframe to shrink in memory footprint size
    :param skip_obj        : If not desired string columns can be skipped during shrink operation
    :param skip_int        : If not desired integer columns can be skipped during shrink operation
    :param skip_float      : If not desired float columns can be skipped during shrink operation
    :param print_comparison: Beware! Printing comparison needs calculation of each columns datasize
                             so if you need speed turn this off. It's just here to show you info                            
    :return                : Pandas Dataframe of exactly the same data and dtypes but in less memory footprint    
    '''
    if print_comparison:
        print(f"Dataframe size before shrinking column types into smallest possible: {round((sys.getsizeof(df)/1024/1024),4)} MB")
    for column in df.columns:
        if (skip_obj is False) and (str(df[column].dtype)[:6] == 'object'):
            num_unique_values = len(df[column].unique())
            num_total_values = len(df[column])
            if num_unique_values / num_total_values < 0.5:
                df.loc[:,column] = df[column].astype('category')
            else:
                df.loc[:,column] = df[column]
        elif (skip_int is False) and (str(df[column].dtype)[:3] == 'int'):
            if df[column].min() > np.iinfo(np.int8).min and df[column].max() < np.iinfo(np.int8).max:
                df[column] = df[column].astype(np.int8)
            elif df[column].min() > np.iinfo(np.int16).min and df[column].max() < np.iinfo(np.int16).max:
                df[column] = df[column].astype(np.int16)
            elif df[column].min() > np.iinfo(np.int32).min and df[column].max() < np.iinfo(np.int32).max:
                df[column] = df[column].astype(np.int32)
        elif (skip_float is False) and (str(df[column].dtype)[:5] == 'float'):
            if df[column].min() > np.finfo(np.float16).min and df[column].max() < np.finfo(np.float16).max:
                df[column] = df[column].astype(np.float16)
            elif df[column].min() > np.finfo(np.float32).min and df[column].max() < np.finfo(np.float32).max:
                df[column] = df[column].astype(np.float32)
    if print_comparison:
        print(f"Dataframe size after shrinking column types into smallest possible: {round((sys.getsizeof(df)/1024/1024),4)} MB")
    return df
# Universal pandas dataframe null/nan cleaner
def df_null_cleaner(df, fill_with=None, drop_na=False, axis=0):
    '''
    Very good information on dealing with missing values of dataframes can be found at 
    http://pandas.pydata.org/pandas-docs/stable/missing_data.html
    
    :param df        : Pandas Dataframe to clean from missing values 
    :param fill_with : Fill missing values with a value of users choice
    :param drop_na   : Drop either axis=0 for rows containing missing fields
                       or axis=1 to drop columns having missing fields default rows                   
    :return          : Pandas Dataframe cleaned from missing values 
    '''
    df[(df == np.NINF)] = np.NaN
    df[(df == np.Inf)] = np.NaN
    if drop_na:
        df.dropna(axis=axis,inplace=True)
    if ~fill_with:
        df.fillna(fill_with, inplace=True)
    return df
def feature_engineering(df,is_train=True):
    if is_train:          
        df = df[df['maxPlace'] > 1].copy()

    target = 'winPlacePerc'
    print('Grouping similar match types together')
    df.loc[(df['matchType'] == 'solo'), 'matchType'] = 1
    df.loc[(df['matchType'] == 'normal-solo'), 'matchType'] = 1
    df.loc[(df['matchType'] == 'solo-fpp'), 'matchType'] = 1
    df.loc[(df['matchType'] == 'normal-solo-fpp'), 'matchType'] = 1

    df.loc[(df['matchType'] == 'duo'), 'matchType'] = 2
    df.loc[(df['matchType'] == 'normal-duo'), 'matchType'] = 2
    df.loc[(df['matchType'] == 'duo-fpp'), 'matchType'] = 2    
    df.loc[(df['matchType'] == 'normal-duo-fpp'), 'matchType'] = 2

    df.loc[(df['matchType'] == 'squad'), 'matchType'] = 3
    df.loc[(df['matchType'] == 'normal-squad'), 'matchType'] = 3    
    df.loc[(df['matchType'] == 'squad-fpp'), 'matchType'] = 3
    df.loc[(df['matchType'] == 'normal-squad-fpp'), 'matchType'] = 3
    
    df.loc[(df['matchType'] == 'flaretpp'), 'matchType'] = 0
    df.loc[(df['matchType'] == 'flarefpp'), 'matchType'] = 0
    df.loc[(df['matchType'] == 'crashtpp'), 'matchType'] = 0
    df.loc[(df['matchType'] == 'crashfpp'), 'matchType'] = 0
    df.loc[(df['rankPoints'] < 0), 'rankPoints'] = 0
    
    print('Adding new features using existing ones')
    df['headshotrate'] = df['kills']/df['headshotKills']
    df['killStreakrate'] = df['killStreaks']/df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df['skill'] = df['headshotKills'] + df['roadKills']
    
    print('Adding normalized features')
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    gc.collect()
    df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
    df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
    df['maxPlaceNorm'] = df['maxPlace']*((100-df['playersJoined'])/100 + 1)
    df['matchDurationNorm'] = df['matchDuration']*((100-df['playersJoined'])/100 + 1)
    df['headshotKillsNorm'] = df['headshotKills']*((100-df['playersJoined'])/100 + 1)
    df['killPlaceNorm'] = df['killPlace']*((100-df['playersJoined'])/100 + 1)
    df['killPointsNorm'] = df['killPoints']*((100-df['playersJoined'])/100 + 1)
    df['killStreaksNorm'] = df['killStreaks']*((100-df['playersJoined'])/100 + 1)
    df['longestKillNorm'] = df['longestKill']*((100-df['playersJoined'])/100 + 1)
    df['roadKillsNorm'] = df['roadKills']*((100-df['playersJoined'])/100 + 1)
    df['teamKillsNorm'] = df['teamKills']*((100-df['playersJoined'])/100 + 1)
    df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
    df['DBNOsNorm'] = df['DBNOs']*((100-df['playersJoined'])/100 + 1)
    df['revivesNorm'] = df['revives']*((100-df['playersJoined'])/100 + 1)    
    
    # Clean null values from dataframe
    df = df_null_cleaner(df,fill_with=0)

    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")
    features.remove("maxPlace")
    
    y = pd.DataFrame()
    if is_train: 
        print('Preparing target variable')
        y = df.groupby(['matchId','groupId'])[target].agg('mean')
        gc.collect()
        features.remove(target)
        
    print('Aggregating means')
    means_features = list(df.columns)
    means_features.remove("Id")
    means_features.remove("matchId")
    means_features.remove("groupId")
    means_features.remove("matchType")  
    means_features.remove("maxPlace")
    means_features.remove("playersJoined")
    means_features.remove("matchDuration")
    means_features.remove("numGroups")
    means_features.remove("teamKillsNorm")
    
    if is_train:
        means_features.remove(target)
    
    agg = df.groupby(['matchId','groupId'])[means_features].agg('mean')
    gc.collect()
    agg_rank = agg.groupby('matchId')[means_features].rank(pct=True).reset_index()
    gc.collect()
    
    if is_train: 
        X = agg.reset_index()[['matchId','groupId']]
    else: 
        X = df[['matchId','groupId']]

    X = X.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    X = X.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    del agg, agg_rank
    gc.collect()
    
    print('Aggregating maxes')
    maxes_features = list(df.columns) 
    maxes_features.remove("Id")
    maxes_features.remove("matchId")
    maxes_features.remove("groupId")
    maxes_features.remove("matchType")  
    maxes_features.remove("DBNOsNorm")
    maxes_features.remove("damageDealtNorm")
    maxes_features.remove("headshotKillsNorm")
    maxes_features.remove("killPlaceNorm")
    maxes_features.remove("killPlace_over_maxPlace")
    maxes_features.remove("killPointsNorm")
    maxes_features.remove("killStreaksNorm")
    maxes_features.remove("killsNorm")
    maxes_features.remove("longestKillNorm")
    maxes_features.remove("matchDurationNorm")
    maxes_features.remove("matchDuration")
    maxes_features.remove("maxPlaceNorm")
    maxes_features.remove("maxPlace")
    maxes_features.remove("numGroups")
    maxes_features.remove("playersJoined")
    maxes_features.remove("revivesNorm")
    maxes_features.remove("roadKillsNorm")
    maxes_features.remove("teamKillsNorm")

    if is_train:
        maxes_features.remove(target)
    
    agg = df.groupby(['matchId','groupId'])[maxes_features].agg('max')
    gc.collect()
    agg_rank = agg.groupby('matchId')[maxes_features].rank(pct=True).reset_index()
    gc.collect()
    X = X.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    X = X.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    del agg, agg_rank
    gc.collect()
    
    print('Aggregating mins')
    mins_features = list(df.columns) 
    mins_features.remove("Id")
    mins_features.remove("matchId")
    mins_features.remove("groupId")
    mins_features.remove("matchType")  
    mins_features.remove("DBNOsNorm")
    mins_features.remove("damageDealtNorm")
    mins_features.remove("headshotKillsNorm")
    mins_features.remove("killPlaceNorm")
    mins_features.remove("killPlace_over_maxPlace")
    mins_features.remove("killPointsNorm")
    mins_features.remove("killStreaksNorm")
    mins_features.remove("killsNorm")
    mins_features.remove("longestKillNorm")
    mins_features.remove("matchDurationNorm")
    mins_features.remove("matchDuration")
    mins_features.remove("maxPlaceNorm")
    mins_features.remove("maxPlace")
    mins_features.remove("numGroups")
    mins_features.remove("playersJoined")
    mins_features.remove("revivesNorm")
    mins_features.remove("roadKillsNorm")
    mins_features.remove("teamKillsNorm")
    
    if is_train:
        mins_features.remove(target)
    
    agg = df.groupby(['matchId','groupId'])[mins_features].agg('min')
    gc.collect()
    agg_rank = agg.groupby('matchId')[mins_features].rank(pct=True).reset_index()
    gc.collect()
    X = X.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    X = X.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    del agg, agg_rank
    gc.collect()
    
    print('Aggregating group sizes')
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    gc.collect()
    X = X.merge(agg, how='left', on=['matchId', 'groupId'])
    print('Aggregating match means')
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    gc.collect()
    X = X.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    print('Aggregating match sizes')
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    gc.collect()
    X = X.merge(agg, how='left', on=['matchId'])
    del df, agg
    gc.collect()

    X.drop(columns = ['matchId', 
                      'groupId'
                     ], axis=1, inplace=True)  
    gc.collect()
    if is_train:
        return X, y
    
    return X
X_train = pd.read_csv('../input/train_V2.csv', engine='c')
X_train = df_footprint_reduce(X_train, skip_obj=True)  # Reduce memory footprint inorder to fit in memory of Kaggle Docker image
gc.collect()
X_train, y_train = feature_engineering(X_train, True)
gc.collect()
X_train = df_footprint_reduce(X_train, skip_obj=True) # Reduce memory footprint again after feature generation
gc.collect()
# Import good old friend
from sklearn.model_selection import train_test_split, GridSearchCV
# Split dataset into train and validation set from %80 of x_train
X_train, X_validation, y_train, y_validation = train_test_split(X_train, 
                                                                y_train, 
                                                                test_size=0.2)
gc.collect()
# Import the real deal
import lightgbm as lgb
# Initialize model with initial parameters given
parameters = { 'objective': 'regression_l1',
               'learning_rate': 0.01
               #'bagging_fraction': 0.6,
               #'bagging_seed': 0,
               #'feature_fraction': 0.8
             }
def find_best_hyperparameters(model):
    # Grid parameters for using in Gridsearch while tuning
    gridParams = {
        'learning_rate'         : [0.1, 0.01 , 0.05],
        'n_estimators '         : [1000, 10000, 20000],
        'bagging_fraction'      : [0.5, 0.6 ,0.7],
        'feature_fraction'      : [0.5, 0.6 ,0.7],
        'num_leaves'            : [31, 80, 140]
    }
    # Create the grid
    grid = GridSearchCV(model, 
                        gridParams,
                        verbose=5,
                        cv=3)
    # Run the grid
    grid.fit(X_train, y_train)
    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)
    return
#find_best_hyperparameters(model)   # This takes time so comment out after finding your right parameters for model training
X_train =  lgb.Dataset(X_train, label=y_train)
X_validation =  lgb.Dataset(X_validation, label=y_validation)
gc.collect()
%%time
# Train model with initial parameters given in initialize section
model = lgb.train(parameters, 
                  X_train,
                  num_boost_round = 40000,
                  valid_sets=[X_validation,X_train])
# Competition evaluation is based on mean absolute error so we calculate it over predictions from test data labels
print('The mean absolute error of model on validation set is:', model.best_score['valid_0']['l1'])
import matplotlib.pyplot as plt
import seaborn as sns
# feature importances
feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(iteration=model.best_iteration),model.feature_name())), 
                           columns=['Value','Feature'])

plt.figure(figsize=(20, 50))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Features')
plt.tight_layout()
plt.show()
# We can eliminate all of these columns because they have no importance to our model
feature_imp[feature_imp['Value']==0]
# Clean memory and load test set
del X_train, X_validation, y_train, y_validation, feature_imp 
gc.collect()
test_x = pd.read_csv('../input/test_V2.csv', engine='c')
test_x = df_footprint_reduce(test_x, skip_obj=True)
gc.collect()
test_x = feature_engineering(test_x, False)
gc.collect()
pred_test = model.predict(test_x, num_iteration=model.best_iteration)
del test_x
gc.collect()
test_set = pd.read_csv('../input/test_V2.csv', engine='c')
submission = pd.read_csv("../input/sample_submission_V2.csv")
submission['winPlacePerc'] = pred_test
submission.loc[submission.winPlacePerc < 0, "winPlacePerc"] = 0
submission.loc[submission.winPlacePerc > 1, "winPlacePerc"] = 1
submission = submission.merge(test_set[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")
submission_group = submission.groupby(["matchId", "groupId"]).first().reset_index()
submission_group["rank"] = submission_group.groupby(["matchId"])["winPlacePerc"].rank()
submission_group = submission_group.merge(
    submission_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
submission_group["adjusted_perc"] = (submission_group["rank"] - 1) / (submission_group["numGroups"] - 1)
submission = submission.merge(submission_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
submission["winPlacePerc"] = submission["adjusted_perc"]
submission.loc[submission.maxPlace == 0, "winPlacePerc"] = 0
submission.loc[submission.maxPlace == 1, "winPlacePerc"] = 1
subset = submission.loc[submission.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
submission.loc[submission.maxPlace > 1, "winPlacePerc"] = new_perc
submission.loc[(submission.maxPlace > 1) & (submission.numGroups == 1), "winPlacePerc"] = 0
assert submission["winPlacePerc"].isnull().sum() == 0
submission[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)