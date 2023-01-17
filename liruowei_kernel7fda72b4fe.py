import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Import dataset

train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
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

train=reduce_mem_usage(train)

test=reduce_mem_usage(test)
train['totalDistance']=train['rideDistance']+train['walkDistance']+train['swimDistance']

train['killsWithoutMoving']=((train['kills']>0)&(train['totalDistance']==0))

train['headshot_rate']=train['headshotKills']/train['kills']

# Add a feature containing the number of players that joined each match.

train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')



print(train.shape)

train.drop(train[train['winPlacePerc'].isnull()].index,inplace=True)

train.drop(train[train['killsWithoutMoving']==True].index,inplace=True)

train.drop(train[train['roadKills']>10].index,inplace=True)

train.drop(train[train['kills']>35].index,inplace=True)

train.drop(train[((train['headshot_rate']==1) & (train['kills']>9))].index,inplace=True)

train.drop(train[train['longestKill']>=1000].index,inplace=True)

train.drop(train[train['walkDistance']>=10000].index,inplace=True)

train.drop(train[train['rideDistance']>=20000].index,inplace=True)

train.drop(train[train['swimDistance']>=2000].index,inplace=True)

train.drop(train[train['weaponsAcquired']>=80].index,inplace=True)

train.drop(train[train['heals']>=40].index,inplace=True)

train.drop(train[( (train['totalDistance']==0) & (train['weaponsAcquired']>3) )].index,inplace=True)

print(train.shape)
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

# Features to remove

train = train.drop([ 'kills', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks', 

 'longestKill', 'roadKills', 'teamKills', 'damageDealt', 'DBNOs', 'revives'],axis=1)
train.head()
target='winPlacePerc'

features=list(train.columns)

features.remove("Id")

features.remove("matchId")

features.remove("groupId")

features.remove("matchType")

features.remove("killsWithoutMoving")

features.remove("headshot_rate")

features.remove(target)

sample = 500000

df_sample = train.sample(sample)

x_train=df_sample[features]

y_train=df_sample[target]
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

random_seed=1

sample = 500000



x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=random_seed)

model=RandomForestRegressor(n_estimators=70,min_samples_leaf=3,max_features=0.5,n_jobs=-1)
%%time

model.fit(x_train,y_train)
print('mae train:',mean_absolute_error(model.predict(x_train),y_train))

print('mae train:',mean_absolute_error(model.predict(x_val),y_val))
test['totalDistance']=test['rideDistance']+test['walkDistance']+test['swimDistance']

#train['killsWithoutMoving']=((train['kills']>0)&(train['totalDistance']==0))

test['headshot_rate']=test['headshotKills']/test['kills']



test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)

test['headshotKillsNorm'] = test['headshotKills']*((100-test['playersJoined'])/100 + 1)

test['killPlaceNorm'] = test['killPlace']*((100-test['playersJoined'])/100 + 1)

test['killPointsNorm'] = test['killPoints']*((100-test['playersJoined'])/100 + 1)

test['killStreaksNorm'] = test['killStreaks']*((100-test['playersJoined'])/100 + 1)

test['longestKillNorm'] = test['longestKill']*((100-test['playersJoined'])/100 + 1)

test['roadKillsNorm'] = test['roadKills']*((100-test['playersJoined'])/100 + 1)

test['teamKillsNorm'] = test['teamKills']*((100-test['playersJoined'])/100 + 1)

test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)

test['DBNOsNorm'] = test['DBNOs']*((100-test['playersJoined'])/100 + 1)

test['revivesNorm'] = test['revives']*((100-test['playersJoined'])/100 + 1)

# Features to remove

test = test.drop([ 'kills', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks', 

 'longestKill', 'roadKills', 'teamKills', 'damageDealt', 'DBNOs', 'revives'],axis=1)
x_test=test[features]

pred=model.predict(x_test)

test['winPlacePerc']=pred

submission=test[['Id','winPlacePerc']]

submission.head()
submission.to_csv('submission.csv',index=False)