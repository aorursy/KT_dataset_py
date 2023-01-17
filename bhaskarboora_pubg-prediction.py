import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
train=pd.read_csv('../input/train_V2.csv')
test=pd.read_csv('../input/test_V2.csv')
ID=test['Id']
train=train.dropna(axis=0)
y_train=train['winPlacePerc']
train=train.drop(['winPlacePerc'],axis=1)
train["players_Match"] = train.groupby("matchId")["Id"].transform("count")
train["players_Group"] = train.groupby("groupId")["Id"].transform("count")

test["players_Match"] = test.groupby("matchId")["Id"].transform("count")
test["players_Group"] = test.groupby("groupId")["Id"].transform("count")
train['Total_Kills'] = train.groupby('groupId')['kills'].transform('sum')
test['Total_Kills'] = test.groupby('groupId')['kills'].transform('sum')
train['First_Man'] = train.groupby('groupId')['matchDuration'].transform('min')
test['First_Man'] = test.groupby('groupId')['matchDuration'].transform('min')
train['Last_Man'] = train.groupby('groupId')['matchDuration'].transform('max')
test['Last_Man'] = test.groupby('groupId')['matchDuration'].transform('max')
train['Time_Survival'] = train['Last_Man'] - train['First_Man']
test['Time_Survival'] = test['Last_Man'] - test['First_Man']
train['Kill_Percentile'] = train['killPlace'] / (train['maxPlace'] + 1e-9)
test['Kill_Percentile'] = test['killPlace'] / (test['maxPlace'] + 1e-9)
train.drop(["matchId","groupId",'Id','killPoints', 'maxPlace', 'winPoints','vehicleDestroys'],axis=1,inplace=True)
test.drop(["matchId","groupId",'Id','killPoints', 'maxPlace', 'winPoints','vehicleDestroys'],axis=1,inplace=True)
train['Headshot_rate'] = train['kills'] / (train['headshotKills'] + 1e-9)
test['Headshot_rate'] = test['kills'] / (test['headshotKills'] + 1e-9)

train['KillStreak_rate'] = train['killStreaks'] / (train['kills'] + 1e-9)
test['KillStreak_rate'] = test['killStreaks'] / (test['kills'] + 1e-9)
train['Total_Damage'] = train['damageDealt'] + train['teamKills']*100
test['Total_Damage'] = test['damageDealt'] + test['teamKills']*100
train['New']=(train['matchDuration'] < train['matchDuration'].mean() )
test['New']=(test['matchDuration'] < test['matchDuration'].mean() )
train['ProKiller']= (train['headshotKills']/train['kills']+1e-9)
test['ProKiller']= (test['headshotKills']/test['kills']+1e-9)
train['Is_Sniper']=(train['longestKill']>=250)
test['Is_Sniper']=(test['longestKill']>=250)
train['killsOverWalkDistance'] = train['kills'] / (train['walkDistance'] + 1e-9)
test['killsOverWalkDistance'] = test['kills'] / (test['walkDistance'] + 1e-9)
train['Total_Distance'] = (train['rideDistance']+train['swimDistance']+train['walkDistance'])
test['Total_Distance'] = (test['rideDistance']+test['swimDistance']+test['walkDistance'])
train['killsOverDistance'] = train['kills'] / (train['distance'] + 1e-9)
test['killsOverDistance'] = test['kills'] / (test['distance'] + 1e-9)
train['Total_enemies'] = train['playersInMatch'] - train['playersInGroup']
test['Total_enemies'] = test['playersInMatch'] - test['playersInGroup']
train['Team_Spirit'] = train['heals'] + train['revives'] + train['boosts']
test['Team_Spirit'] = test['heals'] + test['revives'] + test['boosts']
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
X_train,X_test,y_train,y_test= train_test_split(train,y_train,test_size=0.3)
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
model.fit(train_pool, eval_set=test_pool, plot=True)
train_mse =(mean_absolute_error(y_train, model.predict(X_train)))
test_mse =(mean_absolute_error(y_test, model.predict(X_test)))
    
print('Train error= ',train_mse)
print('Test error= ',test_mse)

subm = pd.read_csv('../input/sample_submission_V2.csv')
predictions = model.predict(test)

test = pd.read_csv('../input/test_V2.csv')
test['winPlacePerc'] = predictions

test['winPlacePerc'] = test.groupby('groupId')['winPlacePerc'].transform('median')

subm['winPlacePerc'] = test['winPlacePerc']
subm['Id']=ID
subm.to_csv('submission.csv', index = False)