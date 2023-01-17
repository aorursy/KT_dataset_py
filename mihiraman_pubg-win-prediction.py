import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import gc
columns = ['groupId','assists','damageDealt','headshotKills','killPlace','killPoints','kills','killStreaks','matchDuration','matchType','revives','teamKills','winPlacePerc']

columns_test = ['Id','groupId','assists','damageDealt','headshotKills','killPlace','killPoints','kills','killStreaks','matchDuration','matchType','revives','teamKills']

dtypes = {
        'assists'           : 'uint8',
        'damageDealt'       : 'float32',
        'headshotKills'     : 'uint8',
        'killPlace'         : 'uint8',
        'killPoints'        : 'uint16',
        'kills'             : 'uint8',
        'killStreaks'       : 'uint8',
        'matchDuration'     : 'uint16',
        'revives'           : 'uint8',
        'teamKills'         : 'uint8',
        'winPlacePerc'      : 'float16'
}

train = pd.read_csv("../input/train_V2.csv",usecols=columns, dtype=dtypes)
test = pd.read_csv("../input/test_V2.csv",usecols=columns_test, dtype=dtypes)
#train.head()
#train.describe()
#train.describe(include='O')
#train.info()
#test.head()
#test.describe()
#test.describe(include='O')
train.loc[train['winPlacePerc'].isna()]
train.loc[train['groupId']=='12dfbede33f92b']
train.dropna(subset=['winPlacePerc'],inplace=True)
train.info(memory_usage='deep')
print('\n- * - * - * - * - * - * - * - * - * - * -\n')
test.info(memory_usage='deep')
train['groupId'] = train.groupId.astype('category')
train['matchType'] = train.matchType.astype('category')
test['groupId'] = test.groupId.astype('category')
test['matchType'] = test.matchType.astype('category')
train.info(memory_usage='deep')
print('\n- * - * - * - * - * - * - * - * - * - * -\n')
test.info(memory_usage='deep')
#data = train

#del train
#gc.collect()
#uniqueteams = data.groupId.unique().tolist()
#for i in range(len(uniqueteams)):
teamsize = train.groupby(['groupId']).size()
teamsize = teamsize.to_dict()
teamsize
train['groupId'] = train['groupId'].map(teamsize)
train = train.rename(columns={'groupId':'Teamsize'})
train.head()
teamsize_test = test.groupby(['groupId']).size()
teamsize_test = teamsize_test.to_dict()
teamsize_test
test['groupId'] = test['groupId'].map(teamsize_test)
test = test.rename(columns={'groupId':'Teamsize'})
test.head()
train = pd.get_dummies(train, columns = ['matchType'])
test = pd.get_dummies(test, columns = ['matchType'])
y = train['winPlacePerc']
X = train.drop('winPlacePerc',axis=1)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)
import lightgbm as lgb

params={'learning_rate': 0.05,
        'objective':'mae',
        'metric':'mae',
        'num_leaves': 31,
        'verbose': 0,
        'random_state':42,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7
       }
model = lgb.LGBMRegressor(**params, n_estimators=1000)
#model.fit(train_X,train_y)
#pred = model.predict(val_X)

#from sklearn.metrics import mean_absolute_error
#mean_absolute_error(pred,val_y)
model.fit(X,y)
Idlist = test.Id
features = test.drop(['Id'],axis=1)
test_preds = model.predict(features)
output = pd.DataFrame({'Id': Idlist,
                       'winPlacePerc': test_preds})

output.to_csv('submission_1.csv', index=False)
#Phew too tired, Later