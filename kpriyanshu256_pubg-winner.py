import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso

from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler,MinMaxScaler
train=pd.read_csv('../input/train_V2.csv')

test=pd.read_csv('../input/test_V2.csv')

ID=test['Id']
train.isna().sum()
train=train.dropna(axis=0)
y_train=train['winPlacePerc']

train=train.drop(['winPlacePerc'],axis=1)
train["playersInMatch"] = train.groupby("matchId")["Id"].transform("count")

train["playersInGroup"] = train.groupby("groupId")["Id"].transform("count")



test["playersInMatch"] = test.groupby("matchId")["Id"].transform("count")

test["playersInGroup"] = test.groupby("groupId")["Id"].transform("count")
train['TotalKills'] = train.groupby('groupId')['kills'].transform('sum')

test['TotalKills'] = test.groupby('groupId')['kills'].transform('sum')
train['FirstMan'] = train.groupby('groupId')['matchDuration'].transform('min')

test['FirstMan'] = test.groupby('groupId')['matchDuration'].transform('min')
train['LastMan'] = train.groupby('groupId')['matchDuration'].transform('max')

test['LastMan'] = test.groupby('groupId')['matchDuration'].transform('max')
train['Survival'] = train['LastMan'] - train['FirstMan']

test['Survival'] = test['LastMan'] - test['FirstMan']
train['Position'] = train['killPlace'] / (train['maxPlace'] + 1e-9)

test['Position'] = test['killPlace'] / (test['maxPlace'] + 1e-9)
train.drop(["matchId","groupId",'Id','killPoints', 'maxPlace', 'winPoints','vehicleDestroys'],axis=1,inplace=True)

test.drop(["matchId","groupId",'Id','killPoints', 'maxPlace', 'winPoints','vehicleDestroys'],axis=1,inplace=True)
train['headshotrate'] = train['kills'] / (train['headshotKills'] + 1e-9)

test['headshotrate'] = test['kills'] / (test['headshotKills'] + 1e-9)



train['killStreakrate'] = train['killStreaks'] / (train['kills'] + 1e-9)

test['killStreakrate'] = test['killStreaks'] / (test['kills'] + 1e-9)
train['TotalDamage'] = train['damageDealt'] + train['teamKills']*100

test['TotalDamage'] = test['damageDealt'] + test['teamKills']*100
train['Noob']=(train['matchDuration'] < train['matchDuration'].mean() )

test['Noob']=(test['matchDuration'] < train['matchDuration'].mean() )
train['Sniper']=(train['longestKill']>=250)

test['Sniper']=(test['longestKill']>=250)
train['ProAim']= (train['headshotKills']/(train['kills']+1e-9))

test['ProAim']= (test['headshotKills']/(test['kills']+1e-9))
train['distance'] = (train['rideDistance']+train['swimDistance']+train['walkDistance'])

test['distance'] = (test['rideDistance']+test['swimDistance']+test['walkDistance'])

    

train['distance'] = np.log1p(train['distance'])

test['distance'] = np.log1p(test['distance'])
set1=set(i for i in train[(train['kills']>40) & (train['heals']==0)].index.tolist())

set2=set(i for i in train[(train['distance']==0) & (train['kills']>20) ].index.tolist())

set3=set(i for i in train[(train['damageDealt']>4000) & (train['heals']<2)].index.tolist())

set4=set(i for i in train[(train['rideDistance']>25000)].index.tolist())

set5=set(i for i in train[(train['killStreaks']>3) & (train['weaponsAcquired']> 30)].index.tolist())

sets=set1 | set2 | set3 | set4 | set5
len(sets)
train=train.drop(list(sets))

y_train=y_train.drop(list(sets))

train.shape
fpp=['crashfpp','duo-fpp','flare-fpp','normal-duo-fpp','normal-solo-fpp','normal-squad-fpp','solo-fpp','squad-fpp']

train["fpp"] = np.where(train["matchType"].isin(fpp),1,0)

test["fpp"] = np.where(test["matchType"].isin(fpp),1,0)
change={'crashfpp':'crash',

        'crashtpp':'crash',

        'duo':'duo',

        'duo-fpp':'duo',

        'flarefpp':'flare',

        'flaretpp':'flare',

        'normal-duo':'duo',

        'normal-duo-fpp':'duo',

        'normal-solo':'solo',

        'normal-solo-fpp':'solo',

        'normal-squad':'squad',

        'normal-squad-fpp':'squad',

        'solo-fpp':'solo',

        'squad-fpp':'squad',

        'solo':'solo',

        'squad':'squad'

       }

train['matchType']=train['matchType'].map(change)

test['matchType']=test['matchType'].map(change)
modes={'crash':1,

       'duo':2,

       'flare':3,

       'solo':4,

       'squad':5

      }

train['matchType']=train['matchType'].map(modes)

test['matchType']=test['matchType'].map(modes)
d1=pd.get_dummies(train['matchType'])

train=train.drop(['matchType'],axis=1)

train=train.join(d1)

    

d2=pd.get_dummies(test['matchType'])

test=test.drop(['matchType'],axis=1)

test=test.join(d2)

    
scaler = MinMaxScaler()

scaler.fit(train)

train=scaler.transform(train)

test=scaler.transform(test)
df = pd.DataFrame(train)

df.isnull().sum()
X_train,X_test,y_train,y_test= train_test_split(train,y_train,test_size=0.3)
lm = Lasso(alpha=1e-5)

lm.fit(X_train,y_train)
train_mse = (mean_absolute_error(y_train,lm.predict(X_train)))

test_mse = (mean_absolute_error(y_test, lm.predict(X_test)))

train_mse,test_mse
y_train = y_train - lm.predict(X_train)

y_test = y_test - lm.predict(X_test)
from catboost import Pool

train_pool = Pool(X_train, y_train)

test_pool = Pool(X_test, y_test) 
model = CatBoostRegressor(

    iterations=5000,

    depth=10,

    learning_rate=0.1,

    l2_leaf_reg= 2,

    loss_function='RMSE',

    eval_metric='MAE',

    random_strength=0.1,

    bootstrap_type='Bernoulli',

    leaf_estimation_method='Gradient',

    leaf_estimation_iterations=1,

    boosting_type='Plain'

    ,task_type = "GPU"

    ,feature_border_type='GreedyLogSum'

    ,random_seed=1234

)
model.fit(train_pool, eval_set=test_pool)
train_mse =(mean_absolute_error(y_train,lm.predict(X_train) + model.predict(X_train)))

test_mse =(mean_absolute_error(y_test, lm.predict(X_test) + model.predict(X_test)))

    

print('Train error= ',train_mse)

print('Test error= ',test_mse)

subm = pd.read_csv('../input/sample_submission_V2.csv')

predictions = model.predict(test) + lm.predict(test)



test = pd.read_csv('../input/test_V2.csv')

test['winPlacePerc'] = predictions



test['winPlacePerc'] = test.groupby('groupId')['winPlacePerc'].transform('median')



subm['winPlacePerc'] = test['winPlacePerc']

subm['Id']=ID

subm.to_csv('submission.csv', index = False)
