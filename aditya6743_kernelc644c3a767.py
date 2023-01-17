# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import seaborn as sbn

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



import matplotlib.pyplot as plt

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train_V2.csv')

data_test = pd.read_csv('../input/test_V2.csv')

data_train.head()

data_test.head()
data_train.info()
data_test.info()
data_train.isna().sum()
data_test.isna().sum(

)
data_train.shape
data_test.shape
data_train.dropna(inplace=True)
data_train.groupby('matchId')[('matchId')].count()
data_test.groupby('matchId')['matchId'].transform('count')
data_train['Match Played'] = data_train.groupby('matchId')['matchId'].transform('count')

data_test['Match Played'] = data_test.groupby('matchId')['matchId'].transform('count')

plt.subplots(figsize =(10,10))

sbn.countplot(data_train['Match Played'])
data_train = pd.get_dummies(data_train,columns=['matchType'])

data_train.shape
data_test  = pd.get_dummies(data_test,columns= ['matchType'])


data_test.shape
data_train['totalDistance'] = data_train['rideDistance']+data_train['swimDistance'] + data_train['walkDistance']
data_test['totalDistance'] = data_test['rideDistance']+data_test['swimDistance'] + data_test['walkDistance']
data_train.drop(labels=['Id','groupId','matchId'],inplace = True,axis=1)

test_id=data_test['Id']

data_test.drop(labels=['Id','groupId','matchId'],inplace = True,axis=1)
data_train.shape
rdata=data_train.sample(3500000)
y = rdata['winPlacePerc']

X_data = rdata.drop(labels='winPlacePerc',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X_data,y,test_size=.3)
rfr = RandomForestRegressor(n_estimators=35, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)
rfr.fit(X_train,y_train)
y_pred = rfr.predict(X_test)
mean_squared_error(y_pred,y_test)
mean_absolute_error(y_pred,y_test)
new_data =pd.DataFrame(sorted(zip(rfr.feature_importances_, X_data.columns)),columns=['Value','Feature'])
new_data # holdiing the column name and its importance value
new_data = new_data.sort_values(by='Value',ascending=False)[:25]
new_data.shape


cols=new_data.Feature.values

X_train,X_test,y_train,y_test = train_test_split(X_data[cols],y,test_size=.3)
rfr = RandomForestRegressor(n_estimators=25, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)
rfr.fit(X_train,y_train)
y_pred = rfr.predict(X_test)
mean_squared_error(y_pred,y_test)
mean_absolute_error(y_pred,y_test)
out = rfr.predict(data_test[cols])
outdf = pd.DataFrame(data = out,columns=['winPlacePerc'])
submisson_V2 = pd.concat([test_id,outdf],axis=1)