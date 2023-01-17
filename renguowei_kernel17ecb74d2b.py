# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
train.drop(2744604,inplace=True)
all_data = pd.concat([train, test])
all_data['playersJoined'] = all_data.groupby('matchId')['matchId'].transform('count')
all_data.drop(columns = ['Id','groupId','matchId'],inplace=True)
all_data['killsNorm'] = all_data['kills']*((100-all_data['playersJoined'])/100+1)

all_data['damageDealtNorm'] = all_data['damageDealt']*((100-all_data['playersJoined'])/100+1)

all_data['totalDistance'] = all_data['rideDistance'] + all_data['walkDistance'] + all_data['swimDistance']

all_data['headshot_rate'] = all_data['headshotKills']/all_data['kills']

all_data['headshot_rate'] = all_data['headshot_rate'].fillna(0)

all_data = pd.get_dummies(all_data,columns=['matchType'])

all_data['BoostAndHeal'] = all_data['boosts'] + all_data['heals']

all_data['skill'] = all_data['roadKills'] +all_data['headshotKills']

all_data['teamwork'] = all_data['revives'] +all_data['assists']
train = all_data.iloc[:4446965]

test = all_data.iloc[4446965:]
train.drop(train[(train['kills']>1)&(train['totalDistance']==0)].index,inplace=True)
train.drop(train[train['kills']>20].index,inplace=True)
X = train.drop(columns=['winPlacePerc'])

Y = train['winPlacePerc']
#sample = 1000000

#df_sample = train.sample(sample)

#df_sample.shape
#y = df_sample['winPlacePerc']

#df = df_sample.drop(columns=['winPlacePerc'])
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from IPython.display import display

from sklearn import metrics

import lightgbm as lgb

from sklearn.metrics import accuracy_score
seed = 7

test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
d_train = lgb.Dataset(X_train, label=y_train)



params = {}

params['objective'] = 'regression'

params['metric'] = 'mae'
model = lgb.train(params, d_train)
y_pred=model.predict(X_test)
y_pred
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("MAE: {}".format(mae))
d_train_full = lgb.Dataset(X, label=Y)

params = {}

params['objective'] = 'regression'

params['metric'] = 'mae'
model_full = lgb.train(params, d_train_full)
X_submit = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
test.drop(columns=['winPlacePerc'], inplace = True)
y_pred_submit=model_full.predict(test)
submission = pd.concat([X_submit,pd.Series(y_pred_submit, name='winPlacePerc')], axis=1)
submission['pred_winPlacePerc'] = submission.iloc[:,-1]

def adjust_pred(x):

    space = 1/(x.maxPlace-1)

    return round(x.pred_winPlacePerc / space) * space



submission['adj_winPlacePerc'] = adjust_pred(submission)



submission.head()
submission = submission.loc[:,['Id','adj_winPlacePerc']]

submission.columns = ['Id','winPlacePerc']

submission.head()
#X_train,X_valid,y_train,y_valid = train_test_split(df,y,random_state=1)
'''

def print_score(m):

    res= ['mae train',mean_absolute_error(m.predict(X_train),y_train),

         'mae val',mean_absolute_error(m.predict(X_valid),y_valid)]

    print (res)

'''
'''

from sklearn.metrics import mean_absolute_error

m1 = RandomForestRegressor(n_estimators=50,n_jobs=-1)

m1.fit(X_train,y_train)

print_score(m1)

'''
submission.to_csv('submission.csv', index=False)