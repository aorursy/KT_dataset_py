import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import pandas as pd 
import numpy as np 
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error
import xgboost as xgb 

%matplotlib inline
train_data = pd.read_csv('../input/train_V2.csv')
#train_data = pd.read_csv('train_V2.csv')
test_data = pd.read_csv('../input/test_V2.csv')
#test_data = pd.read_csv('test_V2.csv')
new_column_names = []
categories = train_data['matchType'].astype('category').cat.categories

for category in categories:
        new_column_names.append('matchType_'+ str(category))
enc = OneHotEncoder(sparse=False)
onehotkat = enc.fit_transform(train_data['matchType'].values.reshape(-1, 1))

tmp = pd.DataFrame(onehotkat, columns=new_column_names)
train_data = pd.concat([train_data, tmp], axis=1)
train_data = train_data.drop(columns=['Id', 'groupId', 'matchId', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks', 'longestKill', 
                          'maxPlace', 'rankPoints', 'matchType'])

train_X = train_data.drop(columns=['winPlacePerc'])
train_y = train_data['winPlacePerc']

train_X = train_X.drop([2744604])
train_y = train_y.drop([2744604])
enc = OneHotEncoder(sparse=False)
onehotkat = enc.fit_transform(test_data['matchType'].values.reshape(-1, 1))

tmp = pd.DataFrame(onehotkat, columns=new_column_names)
test_data = pd.concat([test_data, tmp], axis=1)
test_X = test_data.drop(columns=['Id', 'groupId', 'matchId', 'boosts', 'damageDealt', 'DBNOs', 
                                 'headshotKills', 'killPlace', 'killPoints', 'killStreaks', 
                                 'longestKill', 'maxPlace', 'rankPoints', 'matchType'])
scaler = StandardScaler()
scaler.fit(train_X)
train_X.iloc[:,:] = scaler.transform(train_X)
scaler = StandardScaler()
scaler.fit(test_X)
test_X.iloc[:,:] = scaler.transform(test_X)
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
model.fit(train_X, train_y)
pred = model.predict(test_X)
submission = pd.DataFrame({'Id': test_data['Id'], 'winPlacePerc': pred}, columns=['Id', 'winPlacePerc'])
submission.to_csv('submission.csv', index=False)