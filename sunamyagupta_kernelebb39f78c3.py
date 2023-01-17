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
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
data=pd.read_csv('../input/train_V2.csv') #Reading training file
y=data['winPlacePerc'] #Target
features=['assists','damageDealt','headshotKills','heals','killPoints','kills','killStreaks','matchDuration','matchType','rankPoints','teamKills','weaponsAcquired','winPoints']
X=data[features] #Trainig attributes
import math
from statistics import mean
y.fillna(0.0000,inplace=True)
d={'squad-fpp':1,'duo-fpp':2,'squad':3,'solo-fpp':4,'duo':5,'solo':6,'normal-squad-fpp':7,'crashfpp':8,'normal-duo-fpp':9,'flaretpp':10,'normal-solo-fpp':11,'flarefpp':12,'normal-squad':13,'crashtpp':14,'normal-solo':15,'normal-duo':16}
X.replace({'matchType':d},inplace=True) #Symbolizing string with int
X['assists'].fillna(mean(X['assists']),inplace=True)

X['damageDealt'].fillna(mean(X['damageDealt']),inplace=True)

X['headshotKills'].fillna(mean(X['headshotKills']),inplace=True)

X['heals'].fillna(mean(X['heals']),inplace=True)

X['killPoints'].fillna(mean(X['killPoints']),inplace=True)

X['kills'].fillna(mean(X['kills']),inplace=True)

X['killStreaks'].fillna(mean(X['killStreaks']),inplace=True)

X['matchDuration'].fillna(mean(X['matchDuration']),inplace=True)

X['matchType'].fillna(mean(X['matchType']),inplace=True)

X['rankPoints'].fillna(mean(X['rankPoints']),inplace=True)

X['teamKills'].fillna(mean(X['teamKills']),inplace=True)

X['weaponsAcquired'].fillna(mean(X['weaponsAcquired']),inplace=True)

X['winPoints'].fillna(mean(X['winPoints']),inplace=True)
model=RandomForestRegressor(random_state=1) #Model
model.fit(X,y) #Fitting the data
y_pred=model.predict(X) #Predicting
val_mae = mean_absolute_error(y_pred,y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
data1=pd.read_csv('../input/test_V2.csv') #Reading test file
f=features=['assists','damageDealt','headshotKills','heals','killPoints','kills','killStreaks','matchDuration','matchType','rankPoints','teamKills','weaponsAcquired','winPoints']
X1=data1[f]
X1.replace({'matchType':d},inplace=True)

X1['heals'].fillna(mean(X1['heals']),inplace=True)

X1['killPoints'].fillna(mean(X1['killPoints']),inplace=True)

X1['kills'].fillna(mean(X1['kills']),inplace=True)

X1['killStreaks'].fillna(mean(X1['killStreaks']),inplace=True)

X1['matchDuration'].fillna(mean(X1['matchDuration']),inplace=True)

X1['matchType'].fillna(mean(X1['matchType']),inplace=True)

X1['rankPoints'].fillna(mean(X1['rankPoints']),inplace=True)

X1['teamKills'].fillna(mean(X1['teamKills']),inplace=True)

X1['weaponsAcquired'].fillna(mean(X1['weaponsAcquired']),inplace=True)

X1['winPoints'].fillna(mean(X1['winPoints']),inplace=True)

X1['damageDealt'].fillna(mean(X1['damageDealt']),inplace=True)

X1['headshotKills'].fillna(mean(X1['headshotKills']),inplace=True)
Y=model.predict(X1) #Predicting using actual test data 
output = pd.DataFrame({'Id': data1.Id,'winPlacePerc': Y}) #Storing the result in csv file

output.to_csv('submission.csv', index=False)