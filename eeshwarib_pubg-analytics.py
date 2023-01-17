# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import itertools

import warnings

warnings.filterwarnings("ignore")
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                #    df[col] = df[col].astype(np.float16)

                #el

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
%%time

train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

train = reduce_mem_usage(train)

test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

test = reduce_mem_usage(test)

test1=reduce_mem_usage(test)

print(train.shape, test.shape)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))



train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[0])



mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

train['matchType'] = train['matchType'].apply(mapper)

train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[1])
corr = train.corr()

f,ax = plt.subplots(figsize=(20, 15))

sns.heatmap(train.corr(), annot=True, fmt= '.1f',ax=ax, cmap="BrBG")

sns.set(font_scale=1.25)

plt.show()

data = train.copy()
print("A total of {} players ({:.4f}%) have won without a single kill!".format(len(data[data['winPlacePerc']==1]), 100*len(data[data['winPlacePerc']==1])/len(train)))

data1 = train[train['damageDealt'] == 0].copy()

print("A total of {} players ({:.4f}%) have won without dealing damage!".format(len(data1[data1['winPlacePerc']==1]), 100*len(data1[data1['winPlacePerc']==1])/len(train)))
data = train.copy()

data = data[data['walkDistance'] < train['walkDistance'].quantile(0.99)]

plt.figure(figsize=(15,10))

plt.title("Walking Distance Distribution",fontsize=15)

sns.distplot(data['walkDistance'])

plt.show()
# f,ax1 = plt.subplots(figsize =(15,8))

# sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=data,color='lime',alpha=0.5)

# plt.xlabel('Count of Vehicle Destroys',fontsize = 16,color='blue')

# plt.ylabel('Win Percentage',fontsize = 16,color='blue')

# plt.title('Vehicle Destroyed/ Win Ratio',fontsize = 20,color='blue')

# plt.grid()

# plt.show()
print("In the game on an average a person uses {:.1f} heal items, 99% of people use {} or less, while the doctor used {}.".format(train['heals'].mean(), train['heals'].quantile(0.99), train['heals'].max()))

print("In the game on an average a person uses {:.1f} boost items, 99% of people use {} or less, while the doctor used {}.".format(train['boosts'].mean(), train['boosts'].quantile(0.99), train['boosts'].max()))
train.drop(train[train['winPlacePerc'].isnull()].index, inplace=True)
# Engineer a new feature _totalDistance

train['_totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
# Engineer _headshot_rate feature --- headshots made per kill

train['_headshot_rate'] = train['headshotKills'] / train['kills']

train['_headshot_rate'] = train['_headshot_rate'].fillna(0)
#Defining some functions for plotting graphs, we will be needing a lot of countplot and distplot

def show_countplot(column):

    plt.figure(figsize=(15,8))

    sns.countplot(data=train, x=column).set_title(column)

    plt.show()

    

def show_distplot(column):

    plt.figure(figsize=(15, 8))

    sns.distplot(train[column], bins=50)

    plt.show()
# List of Hitman who made more than 10 kills and all the kills were done by headshot(perfect kill)

display(train[(train['_headshot_rate'] == 1) & (train['kills'] >=10)].shape)

train[(train['_headshot_rate'] == 1) & (train['kills'] >= 10)].head(10)
# Create feature killsWithoutMoving

train['_killsWithoutMoving'] = ((train['kills'] > 0) & (train['_totalDistance'] == 0))

# Check players who kills without moving

display(train[train['_killsWithoutMoving'] == True].shape)

train[train['_killsWithoutMoving'] == True].head(10)
# Drop longestKill 'fraudsters'

train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
# Drop roadKill 'fraudsters'

train.drop(train[train['roadKills'] > 10].index, inplace=True)
# Drop walking anomalies

train.drop(train[(train['walkDistance'] >= 13000) & (train['kills'] == 0)].index, inplace=True)
# Drop riding anomalies

train.drop(train[(train['rideDistance'] >= 30000) & (train['kills'] == 0)].index, inplace = True)

train.drop(train[(train['walkDistance'] == 0) & (train['rideDistance'] > 0) & (train['kills'] > 0)].index, inplace = True)

train.drop(train[(train['_totalDistance'] == 0)].index, inplace=True)
# Remove outliers

train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
# Remove outliers

train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
# Remove outliers

train.drop(train[train['heals'] >= 40].index, inplace=True)
cleaned_data=train.copy()
cleaned_data = reduce_mem_usage(cleaned_data)
cleaned_data.head()
cleaned_data['_playersJoined'] = cleaned_data.groupby('matchId')['matchId'].transform('count')

data = cleaned_data.copy()

data = data[data['_playersJoined']>49]

plt.figure(figsize=(15,10))

sns.countplot(data['_playersJoined'])

plt.title("Players Joined",fontsize=15)

plt.show()
# Create normalized features

cleaned_data['_killsNorm'] = cleaned_data['kills']*((100-cleaned_data['_playersJoined'])/100 + 1)

cleaned_data['_damageDealtNorm'] = cleaned_data['damageDealt']*((100-cleaned_data['_playersJoined'])/100 + 1)

cleaned_data['_maxPlaceNorm'] = cleaned_data['maxPlace']*((100-cleaned_data['_playersJoined'])/100 + 1)

cleaned_data['_matchDurationNorm'] = cleaned_data['matchDuration']*((100-cleaned_data['_playersJoined'])/100 + 1)

# Compare standard features and normalized features

to_show = ['Id', 'kills','_killsNorm','damageDealt', '_damageDealtNorm', 'maxPlace', '_maxPlaceNorm', 'matchDuration', '_matchDurationNorm']

cleaned_data[to_show][0:11]

match = cleaned_data.groupby('matchId')

cleaned_data['_killsPerc'] = match['kills'].rank(pct=True).values

cleaned_data['_killPlacePerc'] = match['killPlace'].rank(pct=True).values

cleaned_data['_walkDistancePerc'] = match['walkDistance'].rank(pct=True).values

cleaned_data['_damageDealtPerc'] = match['damageDealt'].rank(pct=True).values

cleaned_data['_walkPerc_killsPerc'] = cleaned_data['_walkDistancePerc'] / cleaned_data['_killsPerc']

cleaned_data.head()
corr = cleaned_data[['_killsPerc', '_killPlacePerc','_walkDistancePerc','_damageDealtPerc', '_walkPerc_killsPerc','winPlacePerc']].corr()
plt.figure(figsize=(15,8))

sns.heatmap(

    corr,

    xticklabels=corr.columns.values,

    yticklabels=corr.columns.values,

    annot=True,

    linecolor='white',

    linewidths=0.1,

    cmap="BrBG"

)

plt.show()
agg = cleaned_data.groupby(['groupId']).size().to_frame('players_in_team')

cleaned_data = cleaned_data.merge(agg, how='left', on=['groupId'])

cleaned_data['_healthItems'] = cleaned_data['heals'] + cleaned_data['boosts']

cleaned_data['_headshotKillRate'] = cleaned_data['headshotKills'] / cleaned_data['kills']

cleaned_data['_killPlaceOverMaxPlace'] = cleaned_data['killPlace'] / cleaned_data['maxPlace']

cleaned_data['_killsOverWalkDistance'] = cleaned_data['kills'] / cleaned_data['walkDistance']

cleaned_data['_killsOverDistance'] = cleaned_data['kills'] / cleaned_data['_totalDistance']

cleaned_data['_walkDistancePerSec'] = cleaned_data['walkDistance'] / cleaned_data['matchDuration']

cleaned_data.head()
corr = cleaned_data[['killPlace', 'walkDistance','players_in_team','_healthItems', '_headshotKillRate', '_killPlaceOverMaxPlace', '_killsOverWalkDistance', '_killsOverDistance','_walkDistancePerSec','winPlacePerc']].corr()

plt.figure(figsize=(15,8))

sns.heatmap(

    corr,

    xticklabels=corr.columns.values,

    yticklabels=corr.columns.values,

    annot=True,

    linecolor='white',

    linewidths=0.1,

    cmap="BrBG"

)

plt.show()
cleaned_data.shape
cleaned_data.drop(['_headshotKillRate','_killsOverDistance', '_killsOverWalkDistance', ], axis=1, inplace=True)
cleaned_data.drop(['killPoints','matchDuration','maxPlace','numGroups','rankPoints','roadKills','teamKills','winPoints', '_playersJoined', '_maxPlaceNorm', '_matchDurationNorm', '_killsWithoutMoving'], axis=1, inplace=True)
cleaned_data.columns
cleaned_data.corr()
corr_matrix = cleaned_data.corr().abs()



upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(to_drop)

# Drop features 

# cleaned_data.drop(cleaned_data[to_drop], axis=1)
# test.drop(['_playersJoined'], axis=1, inplace=True)

test.shape
test.head(2)
cols_to_fit = [col for col in cleaned_data.columns]

corr = cleaned_data[cols_to_fit].corr()

f,ax = plt.subplots(figsize=(30, 20))

sns.heatmap(corr, annot=True, fmt= '.1f',ax=ax, cmap="BrBG")

sns.set(font_scale=1.25)

plt.show()
highly_corr=cleaned_data.copy()

highly_corr.columns
X_train = highly_corr[highly_corr['winPlacePerc'].notnull()].reset_index(drop=True)

X_test = highly_corr[highly_corr['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)





Y_train = X_train.pop('winPlacePerc')

X_test_grp = X_test[['matchId','groupId']].copy()

train_matchId = X_train['matchId']



# drop matchId,groupId

X_train.drop(['matchId','groupId','Id'], axis=1, inplace=True)

X_test.drop(['matchId','groupId','Id'], axis=1, inplace=True)



print(X_train.shape, X_test.shape)
# One hot encode matchType

highly_corr = pd.get_dummies(test, columns=['matchType'])



# Take a look at the encoding

matchType_encoding = highly_corr.filter(regex='matchType')

matchType_encoding.head()
# highly_corr = pd.get_dummies(highly_corr, columns=['matchType'])

highly_corr.info()
# highly_corr = pd.get_dummies(test, columns=['matchType'])



# # Take a look at the encoding

# matchType_encoding = highly_corr.filter(regex='matchType')

# matchType_encoding.head()
# Turn groupId and match Id into categorical types

highly_corr['groupId'] = highly_corr['groupId'].astype('category')

highly_corr['matchId'] = highly_corr['matchId'].astype('category')



# Get category coding for groupId and matchID

highly_corr['groupId_cat'] = highly_corr['groupId'].cat.codes

highly_corr['matchId_cat'] = highly_corr['matchId'].cat.codes



# Get rid of old columns

highly_corr.drop(columns=['groupId', 'matchId'], inplace=True)



# Lets take a look at our newly created features

highly_corr[['groupId_cat', 'matchId_cat']].head()
# Drop Id column, because it probably won't be useful for our Machine Learning algorithm,

# because the test set contains different Id's

highly_corr.drop(columns = ['Id'], inplace=True)
highly_corr.columns
# Take sample for debugging and exploration

sample = 500000

df_sample = highly_corr.sample(sample)
# Split sample into training data and target variable

df = df_sample.drop(columns = ['winPoints']) #all columns except target

y = df_sample['winPoints'] # Only target variable
# Function for splitting training and validation data

def split_vals(a, n : int): 

    return a[:n].copy(), a[n:].copy()

val_perc = 0.12 # % to use for validation set

n_valid = int(val_perc * sample) 

n_trn = len(df)-n_valid

# Split data

raw_train, raw_valid = split_vals(df_sample, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)



# Check dimensions of samples

print('Sample train shape: ', X_train.shape, 

      'Sample target shape: ', y_train.shape, 

      'Sample validation shape: ', X_valid.shape)
# Metric used for the PUBG competition (Mean Absolute Error (MAE))

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor



# Function to print the MAE (Mean Absolute Error) score



def print_score(m : RandomForestRegressor):

    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train), 

           'mae val: ', mean_absolute_error(m.predict(X_valid), y_valid)]

    #Score of the training dataset obtained using an out-of-bag estimate.

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
# Train basic model

m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)

m1.fit(X_train, y_train)

print_score(m1)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)
# What are the most predictive features according to our basic random forest model

fi = rf_feat_importance(m1, df); fi[:15]
# Plot a feature importance graph for the 20 most important features

plot1 = fi[:15].plot('cols', 'imp', figsize=(14,6), legend=False, kind = 'barh')

plot1
# Keep only significant features

to_keep = fi[fi.imp>0.0001].cols

print('Significant features: ', len(to_keep))

to_keep
# Make a DataFrame with only significant features

df_keep = df[to_keep].copy()

X_train, X_valid = split_vals(df_keep, n_trn)
# Train model on top features

m2 = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)

m2.fit(X_train, y_train)

print_score(m2)
#Adding same features to test data

agg = test.groupby(['groupId']).size().to_frame('players_in_team')

test = test.merge(agg, how='left', on=['groupId'])

test['_headshot_rate'] = test['headshotKills'] / test['kills']

test['_headshot_rate'] = test['_headshot_rate'].fillna(0)

test['_totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']

test['_playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

test['_killsNorm'] = test['kills']*((100-test['_playersJoined'])/100 + 1)

test['_damageDealtNorm'] = test['damageDealt']*((100-test['_playersJoined'])/100 + 1)

test['_healthItems'] = test['heals'] + test['boosts']

test['killsWithoutMoving'] = ((test['kills'] > 0) & (test['_totalDistance'] == 0))

test['_killPlacePerc'] = test['killPlace'].rank(pct=True).values

test['_killsPerc'] = test['kills'].rank(pct=True).values

test['_walkDistancePerc'] = test['walkDistance'].rank(pct=True).values

test['_walkPerc_killsPerc'] = test['_walkDistancePerc'] / test['_killsPerc']

test['_killPlaceOverMaxPlace'] = test['killPlace'] / test['maxPlace']

test['_killsPerc'] = test['kills'].rank(pct=True).values

test['_walkDistancePerc'] = test['walkDistance'].rank(pct=True).values

test['_walkDistancePerSec'] = test['walkDistance'] / test['matchDuration']



# Turn groupId and match Id into categorical types

test['groupId'] = test['groupId'].astype('category')

test['matchId'] = test['matchId'].astype('category')



# Get category coding for groupId and matchID

test['groupId_cat'] = test['groupId'].cat.codes

test['matchId_cat'] = test['matchId'].cat.codes
test.head()
test.info()
# One hot encode matchType

test = pd.get_dummies(test, columns=['matchType'])



# Take a look at the encoding

matchType_encoding = highly_corr.filter(regex='matchType')

matchType_encoding.head()
test.drop('Id',axis=1,inplace=True)
test.drop(['groupId','matchId'],axis=1,inplace=True)
m3 = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=0.5,

                          n_jobs=-1)

m3.fit(X_train, y_train)

print_score(m3)
# # Remove irrelevant features from the test set

test_pred = test[to_keep].copy()



# Fill NaN with 0 (temporary)

test_pred.fillna(0, inplace=True)

test_pred.head()
predictions = np.clip(a = m3.predict(test_pred), a_min = 0.0, a_max = 1.0)

pred_df = pd.DataFrame({'Id' : test1['Id'], 'winPlacePerc' : predictions})

pred_df

# Create submission file

pred_df.to_csv("submission.csv", index=False)
#workshop by edureka!