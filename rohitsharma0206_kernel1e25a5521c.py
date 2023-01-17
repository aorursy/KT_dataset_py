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
train=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

test=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

train.head()
ID=test['Id']

test.drop(['rankPoints','groupId','matchId','Id','matchType','numGroups'],axis=1,inplace=True)

train.drop(['rankPoints','groupId','matchId','Id','matchType','numGroups'],axis=1,inplace=True)
train.head()
test['damageDealt']=test['damageDealt']/test['damageDealt'].max()

train['damageDealt']=train['damageDealt']/train['damageDealt'].max()
train['killPoints']=train['killPoints']/train['killPoints'].max()

test['killPoints']=test['killPoints']/test['killPoints'].max()
train['longestKill']=train['longestKill']/train['longestKill'].max()

test['longestKill']=test['longestKill']/test['longestKill'].max()
test['matchDuration']=test['matchDuration']/test['matchDuration'].max()

train['matchDuration']=train['matchDuration']/train['matchDuration'].max()
train['walkDistance']=train['walkDistance']/train['walkDistance'].max()

test['walkDistance']=test['walkDistance']/test['walkDistance'].max()
train['winPoints']=train['winPoints']/train['winPoints'].max()

test['winPoints']=test['winPoints']/test['winPoints'].max()
train.drop('swimDistance',axis=1,inplace=True)

test.drop('swimDistance',axis=1,inplace=True)
train.head()
train.corr()
train.drop(['killPoints','maxPlace','roadKills','teamKills','vehicleDestroys','winPoints','killPlace','revives','killStreaks','headshotKills','DBNOs','assists'],inplace=True,axis=1)

test.drop(['killPoints','maxPlace','roadKills','teamKills','vehicleDestroys','winPoints','killPlace','revives','killStreaks','headshotKills','DBNOs','assists'],inplace=True,axis=1)
train.corr()
train.drop('matchDuration',inplace=True,axis=1)

train.fillna(train.mean(),inplace=True)

test.drop('matchDuration',inplace=True,axis=1)

test.fillna(test.mean(),inplace=True)
train.corr()
X=train.drop('winPlacePerc',axis=1)

y=train.winPlacePerc
from sklearn.ensemble import RandomForestRegressor

reg=RandomForestRegressor(max_depth=10)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

reg.fit(X_train,y_train)



    
pred=reg.predict(X_test)

from sklearn.metrics import r2_score

r2_score(y_test, pred)
sub=reg.predict(test)

submission=pd.DataFrame({'Id':ID,'winPlacePerc':sub})

submission.head()
filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)
