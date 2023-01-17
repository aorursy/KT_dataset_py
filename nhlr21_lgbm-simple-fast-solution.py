import pandas as pd

import matplotlib

import pydot

import re

import dask.dataframe as dd



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12



import numpy as np

import seaborn as sns

sns.set()

import sklearn



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor 

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import lightgbm as lgb



import gc

gc.enable()
# Create table for missing data analysis

def draw_missing_data_table(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data
#path of datasets

path_train = '../input/train_V2.csv'

path_test = '../input/test_V2.csv'
#create dataframe for training dataset and print ten first rows as preview

train_df_raw = pd.read_csv(path_train)

train_df_raw.head()
# Compute some basical statistics on the dataset

train_df_raw.describe()
draw_missing_data_table(train_df_raw)
train_df_raw.info()
# Let's plot some histograms on main features to have a previzualisation of some of the data ...

train_df_raw.drop(['Id', 'groupId', 'matchId', 'winPoints', 'rankPoints', 'teamKills', 'vehicleDestroys', 'roadKills', 'swimDistance', 'numGroups'], 1).hist(bins=50, figsize=(50,80), layout=(8, 3))

plt.show()
plt.figure(figsize=(20,15))  

sns.heatmap(train_df_raw.corr(), annot=True, fmt=".2f")

plt.show()
train_df_raw.matchType.unique().tolist()
train_df_raw['matchType'].value_counts()
def preprocess_data(df, with_categorical=False):



    processed_df = df.drop(['Id', 'rankPoints'],  axis=1)

            

    # handle matchType column by creating dummies cols or creating new categorical variable column

    print('-'*5 + ' handling matchType column ' + '-'*5)

    new_matchType_cols = list()

    if with_categorical:

        for mtype in processed_df['matchType']:

            if mtype in ['squad', 'squad-fpp', 'normal-squad-fpp', 'normal-squad', 'flarefpp', 'flaretpp']:

                new_matchType_cols.append([3])

            elif mtype in ['solo', 'solo-fpp', 'normal-solo-fpp', 'normal-solo']:

                new_matchType_cols.append([1])

            else:

                new_matchType_cols.append([2])

        match_df = pd.DataFrame(new_matchType_cols, columns=['matchType'], index=processed_df.index)

        

    else:

        for mtype in processed_df['matchType']:

            if mtype in ['squad', 'squad-fpp', 'normal-squad-fpp', 'normal-squad', 'flarefpp', 'flaretpp']:

                new_matchType_cols.append([1, 0, 0])

            elif mtype in ['solo', 'solo-fpp', 'normal-solo-fpp', 'normal-solo']:

                new_matchType_cols.append([0, 0, 1])

            else:

                new_matchType_cols.append([0, 1, 0])

        match_df = pd.DataFrame(new_matchType_cols, columns=['squad','duo', 'solo'], index=processed_df.index)

        

    processed_df = processed_df.drop(['matchType'],  axis=1)

    

    # create matchSize column

    print('-'*5 + ' create matchSize column ' + '-'*5)

    match_size = processed_df.groupby(['matchId']).size().reset_index(name='matchSize')

    processed_df = processed_df.merge(match_size, how='left', on=['matchId'])

    

    # create teamSize column

    print('-'*5 + ' create teamSize column ' + '-'*5)

    processed_df['combinedId'] = processed_df['matchId'] + processed_df['groupId']

    group_size = processed_df.groupby(['combinedId']).size().reset_index(name='teamSize')

    processed_df = processed_df.merge(group_size, how='left', on=['combinedId'])

    

    # create totalDistance col

    print('-'*5 + ' create totalDistance column ' + '-'*5)

    processed_df['totalDistance'] = processed_df['rideDistance'] + processed_df['walkDistance'] + processed_df['swimDistance']

    #processed_df['headshotRate'] = processed_df['headshotKills'] / processed_df['kills']

    #processed_df['killstreaksRate'] = processed_df['killStreaks'] / processed_df['kills']

    

    processed_df = processed_df.drop(['combinedId', 'matchId', 'groupId'],  axis=1)

    processed_df = processed_df.join(match_df)

    

    # delete low importances features

    processed_df = processed_df.drop(['teamKills', 'vehicleDestroys', 'roadKills', 'swimDistance', 'headshotKills', 'solo', 'duo', 'squad'], 1)



    return processed_df
train_df = preprocess_data(train_df_raw.dropna())

X_train = train_df.drop('winPlacePerc', 1)

y_train = train_df['winPlacePerc']

sc = StandardScaler()

X_train = pd.DataFrame(sc.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns)

X_train.head()
test_df_raw = pd.read_csv(path_test)

# assert there are no missing data as in the train dataframe

draw_missing_data_table(test_df_raw)
# apply the same transformation on test dataset than on train dataset

X_test = preprocess_data(test_df_raw)

X_test = pd.DataFrame(sc.fit_transform(X_test.values), index=X_test.index, columns=X_test.columns)

X_test.head()
# Create and train model on train data sample

params = {

    'num_leaves': 2048,

    'learning_rate': 0.01,

    'n_estimators': 1000,

    #'max_depth':10,

    'min_data_in_leaf': 400,

    'max_bin': 1000,

    #'bagging_fraction':0.8,

    #'bagging_freq':5,

    #'feature_fraction':0.9,

    #'verbose':50,

    #'boosting_type': 'rf',

    'random_state': 42,

    'objective' : 'regression',

    'metric': 'mae'

    }



model = lgb.LGBMRegressor(**params, verbose=2, n_jobs=4, silent=False)

model.fit(X_train, y_train, eval_metric= 'mae')
# Predict for test data sample

prediction = model.predict(X_test)
lgb.plot_importance(model)

plt.show()
# Tip found here: https://www.kaggle.com/anycode/simple-nn-baseline-3

for i in range(len(test_df_raw)):

    winPlacePerc = prediction[i]

    maxPlace = int(test_df_raw.iloc[i]['maxPlace'])

    if maxPlace == 0:

        winPlacePerc = 0.0

    elif maxPlace == 1:

        winPlacePerc = 1.0

    else:

        gap = 1.0 / (maxPlace - 1)

        winPlacePerc = round(winPlacePerc / gap) * gap

    

    if winPlacePerc < 0: winPlacePerc = 0.0

    if winPlacePerc > 1: winPlacePerc = 1.0    

    prediction[i] = winPlacePerc
result_df = test_df_raw.copy()

result_df['winPlacePerc'] = prediction



result_df.head()

result_df.to_csv('submission.csv', columns=['Id', 'winPlacePerc'], index=False)