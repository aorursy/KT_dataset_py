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
# using IsolationForest to predict the players who use cheaters
#Because of the low ability of my computer ,i just choose 40000 of the trianing data to run my code
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
train_PUBG = pd.read_csv('../input/train40000items/train40000items.csv')
import pandas as pd
from sklearn.ensemble import IsolationForest
ilf = IsolationForest(n_estimators=100,   
                      n_jobs=-1,         
                      verbose=2,
                      contamination=0.0015#Set the number of possible outliers,in my point of view, it is about 0.0015（perhaps it's higher than what I thought）
    )
train_PUBG=train_PUBG.fillna(0)
# trian
train_PUBG[['assists','boosts','damageDealt','DBNOs','headshotKills','heals','killPlace','killPoints','kills','killStreaks','longestKill','matchDuration','maxPlace','numGroups','rankPoints','revives','rideDistance','roadKills','swimDistance','teamKills','vehicleDestroys','walkDistance','weaponsAcquired','winPoints','winPlacePerc']].astype('float64')
X_cols=['assists','boosts','damageDealt','DBNOs','headshotKills','heals','killPlace','killPoints','kills','killStreaks','longestKill','matchDuration','maxPlace','numGroups','rankPoints','revives','rideDistance','roadKills','swimDistance','teamKills','vehicleDestroys','walkDistance','weaponsAcquired','winPoints','winPlacePerc']

ilf.fit(train_PUBG[X_cols])
pred = ilf.predict(train_PUBG[X_cols])
train_PUBG['pred'] = pred
sns.distplot(train_PUBG.loc[train_PUBG['pred']!=1]['winPlacePerc'])
# this is the Distribution of the cheaters 
sns.distplot(train_PUBG.loc[train_PUBG['pred']==1]['winPlacePerc'])
# this is the Distribution of the normal player 