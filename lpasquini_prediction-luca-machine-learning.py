# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

columns = ['Id','groupId','assists','damageDealt','headshotKills','heals','killPlace','killPoints','kills','killStreaks','matchDuration','matchType','revives','walkDistance','weaponsAcquired','winPlacePerc']



train = pd.read_csv("../input/train_V2.csv",usecols=columns)



#df.head()

train

# Any results you write to the current directory are saved as output.
teamsize = train.groupby(['groupId']).size()

teamsize = teamsize.to_dict()

train['groupId'] = train['groupId'].map(teamsize)

train = train.rename(columns={'groupId':'Teamsize'})

train = pd.get_dummies(train, columns = ['matchType'])

train
train.to_csv('submission.csv', index=False)