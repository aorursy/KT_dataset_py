# For System

import gc



# For DataFrame

import numpy as np # for linear algebra

import pandas as pd # for data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



# For Analysis

from sklearn import neighbors

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from math import exp

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

from sklearn.model_selection import KFold

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression



# For Visualization

import plotly # visualization

from plotly.graph_objs import Scatter, Figure, Layout # visualization

from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot # visualization

import plotly.figure_factory as ff # visualization

import plotly.graph_objs as go # visualization

init_notebook_mode(connected=True) # visualization

import matplotlib.pyplot as plt # for data visualization

import seaborn as sns # for data visualization

color = sns.color_palette()

%matplotlib inline
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_raw = pd.read_csv('../input/train_V2.csv')

test_raw = pd.read_csv('../input/test_V2.csv')


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    #start_mem = df.memory_usage().sum() / 1024**2

    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



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

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    return df

train_raw.head(10)
test_raw.head(10)
# Check dataframe's shape

print('Shape of training set: ', train_raw.shape)

# Types, Data points, memory usage, etc.

print(train_raw.info())





# Check dataframe's shape

print('Shape of test set: ', test_raw.shape)

print(test_raw.info())
train_raw.isnull().sum()
test_raw.isnull().sum()
print("shape of before train drop null data : "+ str(train_raw.shape[0]) + "," + str(train_raw.shape[1]))

train_raw = train_raw.dropna()

print("shape of after train drop null data : "+ str(train_raw.shape[0]) + "," + str(train_raw.shape[1]))
continuous = ['damageDealt','killPlace','killPoints','longestKill','matchDuration','maxPlace','numGroups','rankPoints','rideDistance','walkDistance','winPoints']

discrete = ['assists','boosts','DBNOs','headshotKills','heals','kills','killStreaks','revives','teamKills','vehicleDestroys','weaponsAcquired']

categories = ['matchType']
### Continuous variable plots

'''

for col in continuous:

    values = train_raw[col].dropna()

    lower = np.percentile(values, 1)

    upper = np.percentile(values, 99)

    fig = plt.figure(figsize=(18,9));

    sns.distplot(values[(values>lower) & (values<upper)], color='Sienna', ax = plt.subplot(121));

    sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));

    plt.suptitle(col, fontsize=16)

'''
### Discrete variable plots

'''

NanAsZero = ['']

for col in discrete:

    if col in NanAsZero:

        train_raw[col].fillna(0, inplace=True)

    values = train_raw[col].dropna()  

    fig = plt.figure(figsize=(18,9));

    sns.countplot(x=values, color='Sienna', ax = plt.subplot(121));

    sns.boxplot(y=values, color='Sienna', ax = plt.subplot(122));

    plt.suptitle(col, fontsize=16)

'''
### Categorical variable plots

'''

for col in categories:

    values = train_raw[col].astype('str').value_counts(dropna=False).to_frame().reset_index()

    if len(values) > 30:

        continue

    values.columns = [col, 'counts']

    fig = plt.figure(figsize=(18,9))

    ax = sns.barplot(x=col, y='counts', color='Sienna', data=values, order=values[col]);

    plt.xlabel(col);

    plt.ylabel('Number of occurrences')

    plt.suptitle(col, fontsize=16)



    ### Adding percents over bars

    height = [p.get_height() for p in ax.patches]    

    total = sum(height)

    for i, p in enumerate(ax.patches):    

        ax.text(p.get_x()+p.get_width()/2,

                height[i]+total*0.01,

                '{:1.0%}'.format(height[i]/total),

                ha="center")    

'''
### Continuous variable plots

'''

for col in continuous:

    values = train_raw[col].dropna()

    fig = plt.figure(figsize=(18,9));

    sns.jointplot(x=col, y="winPlacePerc", kind="hex", color="#0000FF", data=train_raw)

    plt.suptitle(col, fontsize=16)

'''
### Discrete variable plots

'''

NanAsZero = ['']

for col in discrete:

    if col in NanAsZero:

        train_raw[col].fillna(0, inplace=True)

    values = train_raw[col].dropna()  

    fig = plt.figure(figsize=(18,9));

    sns.jointplot(x=col, y="winPlacePerc", kind="hex", color="#4CB391", data=train_raw)

    plt.suptitle(col, fontsize=16)

'''
train_raw['mvmtDistance'] = train_raw['walkDistance'] + train_raw['rideDistance']+train_raw['swimDistance']



features = list(train_raw.columns)

features.remove("Id")

features.remove("matchId")

features.remove("groupId")

features.remove("matchType")

features.remove("winPlacePerc")
df_team = train_raw.copy()



df_max= train_raw.groupby(['matchId','groupId'])[features].agg('max')

df_team = pd.merge(train_raw, df_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])

df_team = df_team.drop(["assists_max","killPoints_max","headshotKills_max","numGroups_max","revives_max","teamKills_max","roadKills_max","vehicleDestroys_max"], axis=1)





df_rank = df_max.groupby('matchId')[features].rank(pct=True).reset_index()

df_team = pd.merge(df_team, df_rank, suffixes=["", "_maxRank"], how='left', on=['matchId', 'groupId'])

df_team = df_team.drop(["roadKills_maxRank","matchDuration_maxRank","maxPlace_maxRank","numGroups_maxRank"], axis=1)

del df_max

del df_rank

gc.collect()



df_sum = train_raw.groupby(['matchId','groupId'])[features].agg('sum')

df_team = pd.merge(df_team, df_sum.reset_index(), suffixes=["", "_sum"], how='left', on=['matchId', 'groupId'])

df_team = df_team.drop(["assists_sum","killPoints_sum","headshotKills_sum","numGroups_sum","revives_sum","teamKills_sum","roadKills_sum","vehicleDestroys_sum"], axis=1)

del df_sum

gc.collect()
df = train_raw.copy()

df['active'] = df['weaponsAcquired']+df['revives']+df['kills']+df['heals']

fig = plt.figure(figsize=(18,9));

sns.jointplot(x="active", y="winPlacePerc", kind="hex", color="#0000FF", data=df)

plt.suptitle("active", fontsize=16) 

df[['active','winPlacePerc']].corr()
df = train_raw.copy()

df['mvmtDistance'] = df['walkDistance'] + df['rideDistance']+df['swimDistance']

fig = plt.figure(figsize=(18,9));

sns.jointplot(x="mvmtDistance", y="winPlacePerc", kind="hex", color="#0000FF", data=df)

plt.suptitle("mvmtDistance", fontsize=16) 

df[['mvmtDistance','winPlacePerc']].corr()
df = train_raw.copy()

df = df.loc[~df['matchType'].isin(['normal-solo-fpp', 'solo-fpp', 'solo', 'normal-solo'])]

df['teamPlay'] = (df['assists'] +df['revives'])/(df['teamKills']+1)

fig = plt.figure(figsize=(18,9));

sns.jointplot(x="teamPlay", y="winPlacePerc", kind="hex", color="#0000FF", data=df)

plt.suptitle("teamPlay", fontsize=16) 

df[['teamPlay','winPlacePerc']].corr()
df = train_raw.copy()

df['mvmtDistance'] = df['walkDistance'] + df['rideDistance']+df['swimDistance']

df['headshotPercent'] = (df['damageDealt'])*(df['mvmtDistance'])

df = df.loc[df['headshotPercent']>10]

fig = plt.figure(figsize=(18,9));

sns.jointplot(x="headshotPercent", y="winPlacePerc", kind="hex", color="#0000FF", data=df)

plt.suptitle("headshotPercent", fontsize=16) 

df[['headshotPercent','winPlacePerc']].corr()
df = train_raw.copy()

df['mvmtDistance'] = df['walkDistance'] + df['rideDistance']+df['swimDistance']

normalized = (df['mvmtDistance']-min(df['mvmtDistance']))/(max(df['mvmtDistance'])-min(df['mvmtDistance']))

df['getitemperMvmt'] = (df['weaponsAcquired'] + df['boosts']+df['heals'])/(normalized+1)



fig = plt.figure(figsize=(18,9));

sns.jointplot(x="getitemperMvmt", y="winPlacePerc", kind="hex", color="#0000FF", data=df)

plt.suptitle("getitemperMvmt", fontsize=16) 

df[['getitemperMvmt','winPlacePerc']].corr()
fig = plt.figure(figsize=(18,9));

sns.jointplot(x="mvmtDistance_max", y="winPlacePerc", kind="hex", color="#0000FF", data=df)

plt.suptitle("mvmtDistance_max", fontsize=16) 

print(df[['mvmtDistance_max','winPlacePerc']].corr())



fig = plt.figure(figsize=(18,9));

sns.jointplot(x="damageDealt_maxRank", y="winPlacePerc", kind="hex", color="#0000FF", data=df)

plt.suptitle("damageDealt_maxRank", fontsize=16) 

print(df[['damageDealt_maxRank','winPlacePerc']].corr())
df = train_raw.copy()

df['mvmtDistance'] = df['walkDistance'] + df['rideDistance']+df['swimDistance']

df['distperDuration'] = df['mvmtDistance']/df['matchDuration']



fig = plt.figure(figsize=(18,9));

sns.jointplot(x="mvmtperDuration", y="winPlacePerc", kind="hex", color="#0000FF", data=df)

plt.suptitle("mvmtperDuration", fontsize=16) 

df[['mvmtperDuration','winPlacePerc']].corr()
df_team['mvmtDistance'] = df_team['walkDistance'] + df_team['rideDistance']+df_team['swimDistance']

df_team['damageperMvmt'] = (df_team['mvmtDistance']+1)*df_team['damageDealt_maxRank']



fig = plt.figure(figsize=(18,9));

sns.jointplot(x="damageperMvmt", y="winPlacePerc", kind="hex", color="#0000FF", data=df_team)

plt.suptitle("damageperMvmt", fontsize=16) 

df_team[['mvmtDistance','winPlacePerc']].corr()
fig = plt.figure(figsize=(18,9));

sns.jointplot(x="damageDealt_maxRank", y="winPlacePerc", kind="hex", color="#0000FF", data=df_team)

plt.suptitle("damageDealt_maxRank", fontsize=16) 

df_team[['damageDealt_maxRank','winPlacePerc']].corr()
fig = plt.figure(figsize=(18,9));

sns.jointplot(x="mvmtDistance_maxRank", y="winPlacePerc", kind="hex", color="#0000FF", data=df_team)

plt.suptitle("mvmtDistance_maxRank", fontsize=16) 

df_team[['mvmtDistance_maxRank','winPlacePerc']].corr()
df_team = reduce_mem_usage(df_team)

test_raw = reduce_mem_usage(test_raw)



df_pred = df_team.drop(['Id','groupId','matchId'],axis=1)

df_pred = reduce_mem_usage(df_pred)



df_pred = pd.get_dummies(df_pred)

#,'assists','killPoints','kills','killStreaks','longestKill','matchDuration','maxPlace','numGroups','rankPoints','revives','roadKills','swimDistance','teamKills','vehicleDestroys','winPoints'

df_pred_y = df_pred['winPlacePerc']

df_pred_x = df_pred.drop(['winPlacePerc'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(df_pred_x, df_pred_y, test_size=0.33, random_state=42)
params = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'nthread': -1,

        'verbose': 0,

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.8,

        'subsample_freq': 1,

        'colsample_bytree': 0.6,

        'reg_aplha': 1,

        'reg_lambda': 0.001,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10

    }





train_set = lgb.Dataset(X_train, y_train, silent=True)

model = lgb.train(params, train_set = train_set, num_boost_round=300)

pred_test_y = model.predict(X_test, num_iteration = model.best_iteration)



rms = sqrt(mean_squared_error(y_test, pred_test_y))

rms
rms = sqrt(mean_squared_error(y_test, prediction))

print(rms)



df_result = pd.DataFrame(columns=['PRED','REAL'])

df_result['PRED'] = prediction

df_result['REAL'] = y_test.reset_index(drop = True)

df_result.head(100)
test_raw['mvmtDistance'] = test_raw['walkDistance'] + test_raw['rideDistance']+test_raw['swimDistance']



features = list(test_raw.columns)

features.remove("Id")

features.remove("matchId")

features.remove("groupId")

features.remove("matchType")



df_max= test_raw.groupby(['matchId','groupId'])[features].agg('max')

df_team = pd.merge(test_raw, df_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])

df_team = df_team.drop(["assists_max","killPoints_max","headshotKills_max","numGroups_max","revives_max","teamKills_max","roadKills_max","vehicleDestroys_max"], axis=1)





df_rank = df_max.groupby('matchId')[features].rank(pct=True).reset_index()

df_team = pd.merge(df_team, df_rank, suffixes=["", "_maxRank"], how='left', on=['matchId', 'groupId'])

df_team = df_team.drop(["roadKills_maxRank","matchDuration_maxRank","maxPlace_maxRank","numGroups_maxRank"], axis=1)

del df_max

del df_rank

gc.collect()



df_sum = train_raw.groupby(['matchId','groupId'])[features].agg('sum')

df_team = pd.merge(df_team, df_sum.reset_index(), suffixes=["", "_sum"], how='left', on=['matchId', 'groupId'])

df_team = df_team.drop(["assists_sum","killPoints_sum","headshotKills_sum","numGroups_sum","revives_sum","teamKills_sum","roadKills_sum","vehicleDestroys_sum"], axis=1)

del df_sum

gc.collect()



test = df_team.drop(['Id', 'groupId', 'matchId'],axis=1)

test = pd.get_dummies(test)

pred_test_y = model.predict(test, num_iteration = model.best_iteration)



test = pd.DataFrame(columns=['Id', 'winPlacePerc'])

test['Id'] = test_raw['Id']

test['winPlacePerc'] = pred_test_y



test.to_csv('submission.csv',index=False)