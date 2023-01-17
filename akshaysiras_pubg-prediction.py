import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
train=pd.read_csv("../input/train_V2.csv")

test=pd.read_csv("../input/test_V2.csv")
train.columns
train.head(5)
le=LabelEncoder()
enc=OneHotEncoder()
#LE_out=le.fit(train['matchType'].values)
#train['matchType']=le.transform(train['matchType'].values)
train.loc[(train.matchType!='solo') & (train.matchType!='duo') & (train.matchType!='squad') & (train.matchType!='solo-fpp') & (train.matchType!='duo-fpp') & (train.matchType!='squad-fpp'),'matchType']='other'
#LE_out=le.fit(train['matchType'].values)
#train['matchType']=le.transform(train['matchType'].values)
train['matchType']=train['matchType'].map({'solo':0 , 'duo':1, 'squad':2, 'solo-fpp':3, 'duo-fpp':4, 'squad-fpp':5,'other':6})
train.isnull().sum()
train.dropna(inplace=True)
#train['matchType']=train['matchType'].fillna(0)
train.isnull().sum()
#train['matchType'].isnull().sum()
data=enc.fit(train[['matchType']])
temp=enc.transform(train[['matchType']])
temp1=pd.DataFrame(temp.toarray(),columns=["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp","other"])
temp1=temp1.set_index(train.index.values)
temp1
train=pd.concat([train,temp1],axis=1)


train['killsasist']=train['kills']+train['assists']+train['roadKills']
train['total_distance']=train['swimDistance']+train['rideDistance']+train['walkDistance']
train['external_booster']=train['boosts']+train['weaponsAcquired']+train['heals']
train=train.drop(['assists','kills','swimDistance','rideDistance','walkDistance','boosts','weaponsAcquired','heals','roadKills','rankPoints'],axis=1)
train=train.drop(['killPoints','maxPlace','winPoints'],axis=1)
train['Players_all']=train.groupby('matchId')['Id'].transform('count')
train['players_group']=train.groupby('groupId')['Id'].transform('count')
#len(train.matchId.unique())
train.columns
Y=train.winPlacePerc
train = train.drop(["Id", "groupId", "matchId","winPlacePerc"], axis=1)
del train['matchType']
train.head()
# X_train, X_test, Y_train, Y_test = train_test_split(train,Y, test_size = 0.2)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print("X_train:",X_train.shape)
# print("Y_train:",Y_train.shape)
# print("X_test:",X_test.shape)
# print("Y_test:",Y_test.shape)

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# clf=RandomForestRegressor(n_estimators=20)
# clf.fit(train,Y.values.reshape(-1))

import lightgbm as lgb
d_train = lgb.Dataset(train, label=Y)
params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['sub_feature'] = 0.9
params['num_leaves'] = 511
params['min_data'] = 1
params['max_depth'] = 30
params['min_gain_to_split']= 0.00001
clf = lgb.train(params, d_train,2000)
# lgb_train = lgb.Dataset(X_train, label=Y_train)
# lgb_eval = lgb.Dataset(X_test, Y_test)
# m_lgb = lgb.train(params=lgbm_params, train_set=lgb_train, valid_sets=lgb_eval,early_stopping_rounds=5,num_boost_round=2000)
# import lightgbm as lgb
# d_train = lgb.Dataset(train, label=Y)
# params = {}
# params['boosting_type']: 'gbdt'
# params['objective']: 'regression'
# params['metric']: 'mae'
# params['num_leaves']: 31
# params['learning_rate']: 0.05
# params['feature_fraction']: 0.9
# params['bagging_fraction']: 0.7
# params['num_threads']: 4
# params['bagging_seed']: 0
# params['colsample_bytree']: 0.7
# params['n_estimators']=20000
# params['early_stopping_rounds']:200
# clf = lgb.train(params, d_train,1000)
# import lightgbm as lgb
# lgb_train = lgb.Dataset(X_train, label=Y_train)
# lgb_eval = lgb.Dataset(X_test, Y_test)
# lgbm_params = {'objective': 'regression','metric': 'mae','boosting_type':'gbdt'}
# m_lgb = lgb.train(params=lgbm_params, train_set=lgb_train, valid_sets=lgb_eval,early_stopping_rounds=5,num_boost_round=2000)
# import lightgbm as lgb
# clf = lgb.LGBMRegressor(objective='regression',num_leaves=30,
#                               learning_rate=0.05, n_estimators=720,
#                               max_bin = 20, bagging_fraction = 0.8,
#                               bagging_freq = 5, feature_fraction = 0.8)
# clf.fit(train,Y)
# from sklearn.metrics import r2_score
# Y_pred = m_lgb.predict(X_test, num_iteration=m_lgb.best_iteration)
# r2_score(Y_test,Y_pred)

0.9300739949608487
test.loc[(test.matchType!='solo') & (test.matchType!='duo') & (test.matchType!='squad') & (test.matchType!='solo-fpp') & (test.matchType!='duo-fpp') & (test.matchType!='squad-fpp'),'matchType']='other'
#LE_out=le.fit(train['matchType'].values)
#train['matchType']=le.transform(train['matchType'].values)
test['matchType']=test['matchType'].map({'solo':0 , 'duo':1, 'squad':2, 'solo-fpp':3, 'duo-fpp':4, 'squad-fpp':5,'other':6})
test.isnull().sum()
data_test=enc.fit(test[['matchType']])
temp_test=enc.transform(test[['matchType']])
temp2=pd.DataFrame(temp_test.toarray(),columns=["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp","other"])
temp2=temp2.set_index(test.index.values)
temp2
test=pd.concat([test,temp2],axis=1)
del test['matchType']

test['killsasist']=test['kills']+test['assists']+test['roadKills']
test['total_distance']=test['swimDistance']+test['rideDistance']+test['walkDistance']
test['external_booster']=test['boosts']+test['weaponsAcquired']+test['heals']

test=test.drop(['assists','kills','swimDistance','rideDistance','walkDistance','boosts','weaponsAcquired','heals','roadKills','rankPoints'],axis=1)
test=test.drop(['killPoints','maxPlace','winPoints'],axis=1)
test['Players_all']=test.groupby('matchId')['Id'].transform('count')
test['players_group']=test.groupby('groupId')['Id'].transform('count')
test_id=test.Id
test = test.drop(["Id", "groupId", "matchId"], axis=1)
test.head()
out=clf.predict(test)
submission=pd.DataFrame({'Id':test_id,'winPlacePerc':out})
submission.to_csv('submission.csv', index=False)
