## Something went wrong when importing fastai.structured.

## We fixed this by put the whole source code of fastai.structured in the notebook.

## This was copied from: https://github.com/anandsaha/fastai.part1.v2/blob/master/fastai/structured.py



from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler

from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.ensemble import forest

from sklearn.tree import export_graphviz



def get_sample(df,n):

    """ Gets a random sample of n rows from df, without replacement.

    Parameters:

    -----------

    df: A pandas data frame, that you wish to sample from.

    n: The number of rows you wish to sample.

    Returns:

    --------

    return value: A random sample of n rows of df.

    Examples:

    ---------

    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    >>> get_sample(df, 2)

       col1 col2

    2     3    a

    1     2    b

    """

    idxs = sorted(np.random.permutation(len(df))[:n])

    return df.iloc[idxs].copy()



def proc_df(df, y_fld, skip_flds=None, do_scale=False, na_dict=None,

            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):



    """ proc_df takes a data frame df and splits off the response variable, and

    changes the df into an entirely numeric dataframe.

    Parameters:

    -----------

    df: The data frame you wish to process.

    y_fld: The name of the response variable

    skip_flds: A list of fields that dropped from df.

    do_scale: Standardizes each column in df,Takes Boolean Values(True,False)

    na_dict: a dictionary of na columns to add. Na columns are also added if there

        are any missing values.

    preproc_fn: A function that gets applied to df.

    max_n_cat: The maximum number of categories to break into dummy values, instead

        of integer codes.

    subset: Takes a random subset of size subset from df.

    mapper: If do_scale is set as True, the mapper variable

        calculates the values used for scaling of variables during training time(mean and standard deviation).

    Returns:

    --------

    [x, y, nas, mapper(optional)]:

        x: x is the transformed version of df. x will not have the response variable

            and is entirely numeric.

        y: y is the response variable

        nas: returns a dictionary of which nas it created, and the associated median.

        mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continous

        variables which is then used for scaling of during test-time.

    Examples:

    ---------

    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    note the type of col2 is string

    >>> train_cats(df)

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    now the type of col2 is category { a : 1, b : 2}

    >>> x, y, nas = proc_df(df, 'col1')

    >>> x

       col2

    0     1

    1     2

    2     1

    >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],

                 children=[4., 6, 3, 3, 2, 3, 5, 4],

                 salary=[90, 24, 44, 27, 32, 59, 36, 27])

    >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),

                          ([:children], StandardScaler())])

    >>>round(fit_transform!(mapper, copy(data)), 2)

    8x4 Array{Float64,2}:

    1.0  0.0  0.0   0.21

    0.0  1.0  0.0   1.88

    0.0  1.0  0.0  -0.63

    0.0  0.0  1.0  -0.63

    1.0  0.0  0.0  -1.46

    0.0  1.0  0.0  -0.63

    1.0  0.0  0.0   1.04

    0.0  0.0  1.0   0.21

    """

    if not skip_flds: skip_flds=[]

    if subset: df = get_sample(df,subset)

    df = df.copy()

    if preproc_fn: preproc_fn(df)

    y = df[y_fld].values

    df.drop(skip_flds+[y_fld], axis=1, inplace=True)



    if na_dict is None: na_dict = {}

    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)

    if do_scale: mapper = scale_vars(df, mapper)

    for n,c in df.items(): numericalize(df, c, n, max_n_cat)

    res = [pd.get_dummies(df, dummy_na=True), y, na_dict]

    if do_scale: res = res + [mapper]

    return res



def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)



def set_rf_samples(n):

    """ Changes Scikit learn's random forests to give each tree a random sample of

    n random rows.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n))



def reset_rf_samples():

    """ Undoes the changes produced by set_rf_samples.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n_samples))
# For autoreloading modules

%load_ext autoreload

%autoreload 2

# For notebook plotting

%matplotlib inline



# Standard libraries

import os

import numpy as np

import pandas as pd



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



import seaborn as sns

from pdpbox import pdp

from plotnine import *

from pandas_summary import DataFrameSummary

from IPython.display import display

import pickle



# Machine Learning

import xgboost as xgb

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

from scipy.cluster import hierarchy as hc

from fastai.imports import *
train = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")

test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")
train.head()
train.describe()
train.info()
# Check row with NaN value

train[train['winPlacePerc'].isnull()]
# Let's drop this entity

train.drop(2744604, inplace=True)
# Check row with NaN value

train[train['winPlacePerc'].isnull()]
print("The average person kills {:.4f} players, 99% of people have {} kills or less, while the most kills ever recorded is {}.".format(train['kills'].mean(),train['kills'].quantile(0.99), train['kills'].max()))
train_copy = train.copy()

train_copy.loc[train_copy['kills'] > train_copy['kills'].quantile(0.99)] = '8+'

sns.countplot(train_copy['kills'].astype('str').sort_values())

plt.title("Kill Count", fontsize=15)

plt.show()
train_copy = train.copy()

train_copy = train_copy[train_copy['kills'] == 0]

plt.title("Damage Dealt by 0 killers", fontsize=15)

sns.distplot(train_copy['damageDealt'])

plt.show()
train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']

train['healsAndBoosts'] = train['heals']+train['boosts']
# A column for team

train['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in train['numGroups']]

train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
# Create normalized features

train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)

train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)

train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)

train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)

# Compare standard features and normalized features

to_show = ['Id', 'kills','killsNorm','damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm', 'matchDuration', 'matchDurationNorm']

train[to_show][0:11]
# Create headshot_rate feature

train['headshot_rate'] = train['headshotKills'] / train['kills']

train['headshot_rate'] = train['headshot_rate'].fillna(0)
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

# Got those cheaters

train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
# Players who got more than 10 roadkills

train[train['roadKills'] > 10]
# Drop roadkill 'cheaters'

train.drop(train[train['roadKills'] > 10].index, inplace=True)



#Note that player c3e444f7d1289d drove 5 meters but killed 14 people with it. Sounds insane doesn't it?
#Let's plot the total kills for every player first. It doesn't look like there are too many outliers

# Plot the distribution of kills

plt.figure(figsize=(12,4))

sns.countplot(data=train, x=train['kills']).set_title('Kills')

plt.show()
# Players who got more than 30 kills

display(train[train['kills'] > 30].shape)

train[train['kills'] > 30].head(10)
# Remove outliers

train.drop(train[train['kills'] > 30].index, inplace=True)
# Plot the distribution of headshot_rate

plt.figure(figsize=(12,4))

sns.distplot(train['headshot_rate'], bins=10)

plt.show()
# Players who made a minimum of 10 kills and have a headshot_rate of 100%

display(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].shape)

train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].head(10)
# Plot the distribution of longestKill

plt.figure(figsize=(12,4))

sns.distplot(train['longestKill'], bins=10)

plt.show()
# Check out players who made kills with a distance of more than 1 km

display(train[train['longestKill'] >= 1000].shape)

train[train['longestKill'] >= 1000].head(10)
# Remove outliers

train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
# Summary statistics for the Distance features

train[['walkDistance', 'rideDistance', 'swimDistance', 'totalDistance']].describe()
# walkDistance anomalies

display(train[train['walkDistance'] >= 10000].shape)

train[train['walkDistance'] >= 10000].head(10)
# Remove outliers

train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)
# rideDistance anomalies

display(train[train['rideDistance'] >= 20000].shape)

train[train['rideDistance'] >= 20000].head(10)
# Remove outliers

train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
# Players who swam more than 2 km

train[train['swimDistance'] >= 2000]
# Remove outliers

train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
# Plot the distribution of weaponsAcquired

plt.figure(figsize=(12,4))

sns.distplot(train['weaponsAcquired'], bins=100)

plt.show()
# Players who acquired more than 80 weapons

display(train[train['weaponsAcquired'] >= 80].shape)

train[train['weaponsAcquired'] >= 80].head()
# Remove outliers

train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
# Distribution of heals

plt.figure(figsize=(12,4))

sns.distplot(train['heals'], bins=10)

plt.show()
# 40 or more healing items used

display(train[train['heals'] >= 40].shape)

train[train['heals'] >= 40].head(10)
# Remove outliers

train.drop(train[train['heals'] >= 40].index, inplace=True)
# Remaining players in the training set

train.shape
print('There are {} different Match types in the dataset.'.format(train['matchType'].nunique()))
# One hot encode matchType

train = pd.get_dummies(train, columns=['matchType'])



# Take a look at the encoding

matchType_encoding = train.filter(regex='matchType')

matchType_encoding.head()
# Turn groupId and match Id into categorical types

train['groupId'] = train['groupId'].astype('category')

train['matchId'] = train['matchId'].astype('category')



# Get category coding for groupId and matchID

train['groupId_cat'] = train['groupId'].cat.codes

train['matchId_cat'] = train['matchId'].cat.codes



# Get rid of old columns

train.drop(columns=['groupId', 'matchId'], inplace=True)



# Lets take a look at our newly created features

train[['groupId_cat', 'matchId_cat']].head()
# Drop Id column, because it probably won't be useful for our Machine Learning algorithm,

# because the test set contains different Id's

train.drop(columns = ['Id'], inplace=True)
train.shape
train.info()
# Add engineered features to the test set

test['totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']

test['healsAndBoosts'] = test['heals'] + test['boosts']

test['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in test['numGroups']]

test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)

test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)

test['maxPlaceNorm'] = test['maxPlace']*((100-train['playersJoined'])/100 + 1)

test['matchDurationNorm'] = test['matchDuration']*((100-test['playersJoined'])/100 + 1)

test['headshot_rate'] = test['headshotKills'] / test['kills']

test['headshot_rate'] = test['headshot_rate'].fillna(0)

test['killsWithoutMoving'] = ((test['kills'] > 0) & (test['totalDistance'] == 0))



# One hot encode matchType

test = pd.get_dummies(test, columns=['matchType'])



# Turn groupId and match Id into categorical types

test['groupId'] = test['groupId'].astype('category')

test['matchId'] = test['matchId'].astype('category')



# Get category coding for groupId and matchID

test['groupId_cat'] = test['groupId'].cat.codes

test['matchId_cat'] = test['matchId'].cat.codes



# Get rid of old columns

test.drop(columns=['groupId', 'matchId'], inplace=True)
test.shape
test.info()
# Take sample for debugging and exploration

sample = 500000

train_sample = train.sample(sample)
# Split sample into training data and target variable

X = train_sample.drop(columns=['winPlacePerc']) # All columns except target

y = train_sample['winPlacePerc'] # Only target variable
# Function for splitting training and validation data

def split_vals(a, n : int):

    return a[:n].copy(), a[n:].copy()

val_perc = 0.12 # % to use for validation set

n_valid = int(val_perc * sample)

n_trn = len(X) - n_valid



# Split data

raw_train, raw_valid = split_vals(train_sample, n_trn)

X_train, X_valid = split_vals(X, n_trn)

y_train, y_valid = split_vals(y, n_trn)



# Check dimensions of samples

print('Sample train shape: ', X_train.shape, 

      'Sample target shape: ', y_train.shape, 

      'Sample validation shape: ', X_valid.shape)
# Function to print the MAE (Mean Absolute Error) score

# This is the metric used by Kaggle in this competition

def print_score(m : xgb):

    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train), 

           'mae val: ', mean_absolute_error(m.predict(X_valid), y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
# # Building DMatrix for XGboost, because it does not work on numpy matrix

# dtrain = xgb.DMatrix(X_train, label=y_train)

# dvalid = xgb.DMatrix(X_valid, label=y_valid)
# # "Learn" the mean from the training data

# mean_train = np.mean(y_train)

# # Get predictions on the test set

# baseline_predictions = np.ones(y_valid.shape) * mean_train

# # Compute MAE

# mae_baseline = mean_absolute_error(y_valid, baseline_predictions)

# print("Baseline MAE is {:.4f}".format(mae_baseline))
# params = {'min_child_weight':[4,5], 

#           'gamma':[i/10.0 for i in range(3,6)],  

#           'subsample':[i/10.0 for i in range(6,11)], 

#           'colsample_bytree':[i/10.0 for i in range(6,11)], 

#           'max_depth': [2,3,4]

#          }



# # Number of trees in Random forest

# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# # Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

# max_depth.append(None)

# # Minimum number of samples required at each split node

# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 3]

# # Method for selecting samples for training each tree

# bootstrap = [True, False]



# param_dist_rf = {

#     'n_estimators': n_estimators,

#     'max_features': max_features,

#     'max_depth': max_depth,

#     'min_samples_split': min_samples_split,

#     'min_samples_leaf' : min_samples_leaf,

#     'bootstrap' : bootstrap

# }



# print(param_dist_rf)
# xgb_model = xgb.XGBRegressor(learning_rate=0.02, n_estimators=600, objective='binary:logistic', silent=True, nthread=1)

rnd_mod_1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)

# rnd_reg_1 = RandomForestRegressor()
# random_rf = RandomizedSearchCV(rnd_reg_1, param_distributions=param_dist_rf,

#                               n_iter=25,

#                               cv=5,

#                               scoring='neg_mean_absolute_error',

#                               error_score=0,

#                               n_jobs=-1,

#                               verbose=3)
# xgb_model.fit(X_train, y_train)

# print_score(xgb_model)



rnd_mod_1.fit(X_train, y_train)

print_score(rnd_mod_1)



# random_rf.fit(X_train, y_train)
# print(random_rf.best_params_)
# pkl_rnd_rf = "pickle_rnd_rf.pkl"

# with open(pkl_rnd_rf, 'wb') as file:

#     pickle.dump(random_rf, file)
# fi = rf_feat_importance(xgb_model, X); fi[:10]

fi = rf_feat_importance(rnd_mod_1, X); fi[:10]
# Plot a feature importance graph for the 20 most important features

plot1 = fi[:20].plot('cols', 'imp', figsize=(14,6), legend=False, kind = 'barh')

plot1
# Keep only significant features

to_keep = fi[fi.imp>0.005].cols

print('Significant features: ', len(to_keep))

to_keep
# Make a DataFrame with only significant features

X_keep = X[to_keep].copy()

X_train, X_valid = split_vals(X_keep, n_trn)
# xgb_model_1 = xgb.XGBRegressor(learning_rate=0.02, n_estimators=800, objective='binary:logistic', silent=True, nthread=1)

rnd_mod_2 = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)
# xgb_model_1.fit(X_train, y_train)

# print_score(xgb_model_1)

rnd_mod_2.fit(X_train, y_train)

print_score(rnd_mod_2)
# Get feature importances of our top features

fi_to_keep = rf_feat_importance(rnd_mod_2, X_keep)

plot2 = fi_to_keep.plot('cols', 'imp', figsize=(14,6), legend=False, kind = 'barh')

plot2
# Correlation Heat map

corr = X_keep.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Create heatmap

heatmap = sns.heatmap(corr)
# Prepare data

val_perc_full = 0.12 # % to use for validation set

n_valid_full = int(val_perc_full * len(train)) 

n_trn_full = len(train)-n_valid_full

X_full = train.drop(columns = ['winPlacePerc']) # all columns except target

y = train['winPlacePerc'] # target variable

X_full = X_full[to_keep] # Keep only relevant features

X_train, X_valid = split_vals(X_full, n_trn_full)

y_train, y_valid = split_vals(y, n_trn_full)



# Check dimensions of data

print('Sample train shape: ', X_train.shape, 

      'Sample target shape: ', y_train.shape, 

      'Sample validation shape: ', X_valid.shape)
# xgb_model_2 = xgb.XGBRegressor(learning_rate=0.02, n_estimators=999, objective='binary:logistic', silent=True, nthread=1)

rnd_mod_3 = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
# xgb_model_2.fit(X_train, y_train)

# print_score(xgb_model_2)

rnd_mod_3.fit(X_train, y_train)

print_score(rnd_mod_3)
# # Remove irrelevant features from the test set

test_pred = test[to_keep].copy()



# Fill NaN with 0 (temporary)

test_pred.fillna(0, inplace=True)

test_pred.head()
# Make submission ready for Kaggle

# We use our final Random Forest model (rnd_mod_3) to get the predictions

predictions = np.clip(a = rnd_mod_3.predict(test_pred), a_min = 0.0, a_max = 1.0)

pred_df = pd.DataFrame({'Id' : test['Id'], 'winPlacePerc' : predictions})



# Create submission file

pred_df.to_csv("submission.csv", index=False)