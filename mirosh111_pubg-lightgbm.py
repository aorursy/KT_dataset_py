# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train_V2.csv")
test=pd.read_csv("../input/test_V2.csv")
le=LabelEncoder()
enc=OneHotEncoder()

train.loc[(train.matchType!='solo') & (train.matchType!='duo') & (train.matchType!='squad') & (train.matchType!='solo-fpp') & (train.matchType!='duo-fpp') & (train.matchType!='squad-fpp'),'matchType']='other'

train['matchType']=train['matchType'].map({'solo':0 , 'duo':1, 'squad':2, 'solo-fpp':3, 'duo-fpp':4, 'squad-fpp':5,'other':6})
train.dropna(inplace=True)
train.isnull().sum()
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
Y=train.winPlacePerc
train = train.drop(["Id", "groupId", "matchId","winPlacePerc"], axis=1)
del train['matchType']
train.head()
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
test.loc[(test.matchType!='solo') & (test.matchType!='duo') & (test.matchType!='squad') & (test.matchType!='solo-fpp') & (test.matchType!='duo-fpp') & (test.matchType!='squad-fpp'),'matchType']='other'

test['matchType']=test['matchType'].map({'solo':0 , 'duo':1, 'squad':2, 'solo-fpp':3, 'duo-fpp':4, 'squad-fpp':5,'other':6})
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