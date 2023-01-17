# Author: malai_guhan(ram1510).This is released under the Apache 2.0 open source license.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings('ignore')
train=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
test=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
x=train.corr()
plt.figure(figsize=(20,20))
sns.heatmap(x,annot=True)
sns.jointplot(data=train,y='DBNOs',x='winPlacePerc',height=10,ratio=3)
sns.jointplot(x='winPlacePerc',y='assists',data=train,height=10,ratio=3)
sns.jointplot(x='winPlacePerc',y='revives',data=train,height=10,ratio=3)
mod_train=train[0:100000]
np.unique(train['matchType'])
mio=lambda x:'Solo' if 'solo' in x else 'duo' if ('duo' or 'crash') in x else 'squad'
mod_train['matchtype']=mod_train['matchType'].apply(mio)
train_data=mod_train.drop(columns=['winPlacePerc','matchType','Id','groupId','matchId'])
train_data.info()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
train_data=pd.get_dummies(train_data,drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(train_data,mod_train['winPlacePerc'],random_state=100,test_size=0.2)
classifier=RandomForestRegressor()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test,y_pred)
print(f'The mean absolute error is {mae}')

mod_train.columns
mod_train['totaldistancetravelled']=mod_train['rideDistance']+mod_train['walkDistance']
len(mod_train)
#To find the number of players in a match
no_of_players=train.groupby('matchId')['Id'].size().to_frame('no.of.players')
mod1_data=pd.merge(mod_train,no_of_players,on='matchId',how='inner')
mod1_data['no.of.players'].value_counts()
len(mod1_data)
plt.figure(figsize=(25,25))
sns.heatmap(mod1_data.corr(),annot=True)
#mod1_data.corr()
mod1_data.columns
train_data=mod1_data.drop(columns=['winPlacePerc','matchType','Id','groupId','matchId','killPoints','matchDuration','maxPlace','numGroups','rankPoints','roadKills','teamKills','vehicleDestroys','winPoints'])
train_data=pd.get_dummies(train_data)
x_train,x_test,y_train,y_test=train_test_split(train_data,mod1_data['winPlacePerc'],random_state=100,test_size=0.2)
classifier.fit(x_train,y_train)
x_pred=classifier.predict(x_test)
accuracy=r2_score(y_test,x_pred)
print(f'The accuracy is {accuracy}')
from sklearn.ensemble import GradientBoostingRegressor
classifier1=GradientBoostingRegressor(n_estimators=100)
classifier1.fit(x_train,y_train)
x_pred=classifier1.predict(x_test)
accuracy=r2_score(y_test,x_pred)
print(f'The accuracy is {accuracy}')