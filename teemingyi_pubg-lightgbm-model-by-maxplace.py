# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read the data
data = pd.read_csv('../input/train_V2.csv')
data.shape
data = data.dropna()
data.shape
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
data.matchType.unique()
def merge_matchType(x):
    if x in {'normal-squad-fpp', 'crashfpp', 'crashtpp', 'normal-duo-fpp',
       'flarefpp', 'normal-solo-fpp', 'flaretpp', 'normal-duo',
       'normal-squad', 'normal-solo'}:
        return 'others'
    else:
        return x
data['matchType'] = data.matchType.apply(merge_matchType)

data.matchType.unique()
data_dumm = pd.get_dummies(data, columns=['matchType'])
data_dumm.head()
data_dumm = data_dumm.drop('matchType_others', axis=1)
data_dumm.columns
data = data_dumm.loc[:,['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace', 'numGroups',
       'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance',
       'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',
       'winPoints', 'matchType_duo', 'matchType_duo-fpp',
       'matchType_solo', 'matchType_solo-fpp', 'matchType_squad',
       'matchType_squad-fpp', 'winPlacePerc']]
print(data.shape)
data.head()
data.maxPlace.plot(kind='hist')
print('There are {} unique maxPlace.'.format(len(data.maxPlace.unique())))
data_store_by_maxPlace = {}
for x in data.maxPlace.unique():
    data_store_by_maxPlace[x] = data.loc[data.maxPlace==x]
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4

def adjust_pred(x, maxPlace):
    space = 1/(maxPlace-1)
    return np.round(x / space) * space
def generate_lgb_model(data, for_eval):
    # split data into X and y
    X = data.iloc[:,3:33]
    
    maxPlace = X.maxPlace.unique()
    X = X.drop('maxPlace', axis=1)
    Y = data.iloc[:,33]
    
    if for_eval == True:
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    else:
        X_train, y_train = X, Y
        
    d_train = lgb.Dataset(X_train, label=y_train)

    params = {}
    params['objective'] = 'regression'
    params['metric'] = 'mae'

    model = lgb.train(params, d_train)
    
    if for_eval == True:
        #Prediction
        y_pred=model.predict(X_test)
        
        y_pred = adjust_pred(y_pred, maxPlace)
        mae = mean_absolute_error(y_test, y_pred)
        print("MAE for maxPlace = {}: {}".format(maxPlace, mae))
    
        return [model, (list(y_test), y_pred)]
    
    else:
        return model
model_store = {}
maxPlace_set = set(data_store_by_maxPlace.keys())
print('There are {} unique maxPlace.'.format(len(maxPlace_set)))

for key, value in data_store_by_maxPlace.items():
    model_store[key] = generate_lgb_model(value,for_eval=True)
    maxPlace_set = maxPlace_set - {key}
    print('There are {} more models to go.'.format(len(maxPlace_set)))
y_test_overall = []
y_pred_overall = []
for value in model_store.values():
    y_test_overall.append(value[1][0])
    y_pred_overall.append(value[1][1])
print('The overall mae is {}.'.format(mean_absolute_error([y for x in y_test_overall for y in x],[y for x in y_pred_overall for y in x])))
model_store_full = {}
maxPlace_set = set(data_store_by_maxPlace.keys())
print('There are {} unique maxPlace.'.format(len(maxPlace_set)))

for key, value in data_store_by_maxPlace.items():
    model_store_full[key] = generate_lgb_model(value,for_eval=False)
    maxPlace_set = maxPlace_set - {key}
    print('There are {} more models to go.'.format(len(maxPlace_set)))
X_submit = pd.read_csv('../input/test_V2.csv')
print(X_submit.shape)
X_submit.head()
X_submit['matchType'] = X_submit.matchType.apply(merge_matchType)

X_submit.matchType.unique()

X_submit_dumm = pd.get_dummies(X_submit, columns=['matchType'])
X_submit_dumm.head()

X_submit_dumm = X_submit_dumm.drop('matchType_others', axis=1)

X_submit = X_submit_dumm.loc[:,['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace', 'numGroups',
       'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance',
       'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',
       'winPoints', 'matchType_duo', 'matchType_duo-fpp',
       'matchType_solo', 'matchType_solo-fpp', 'matchType_squad',
       'matchType_squad-fpp']]

X_submit.head()
data_store_by_maxPlace_submit = {}
for x in X_submit.maxPlace.unique():
    data_store_by_maxPlace_submit[x] = X_submit.loc[X_submit.maxPlace==x]
#Prediction
prediction = {}
for key, value in data_store_by_maxPlace_submit.items():
    maxPlace = key
    train_data = value.iloc[:,3:33]

    train_data = train_data.drop('maxPlace', axis=1)
    
    pred_submit = model_store_full[key].predict(train_data)
    
    pred_submit = adjust_pred(pred_submit, maxPlace)
    prediction[key] = pd.concat([value.Id.reset_index(drop=True), pd.Series(pred_submit, name='winPlacePerc')], axis=1)
#Submission file
submission = pd.concat([x for x in prediction.values()])

submission.head()
submission.to_csv('submission.csv', index=False)