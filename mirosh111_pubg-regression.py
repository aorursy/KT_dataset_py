# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import xgboost
from sklearn.ensemble import RandomForestRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train_V2.csv')
test_df = pd.read_csv('../input/test_V2.csv')
matchTyp = {'squad-fpp': 0, 'duo': 1, 'solo-fpp': 2, 'squad': 3, 'duo-fpp': 4, 'solo': 5,
       'normal-squad-fpp': 6, 'crashfpp': 7, 'flaretpp': 8, 'normal-solo-fpp': 9,
       'flarefpp': 10, 'normal-duo-fpp': 11, 'normal-duo': 12, 'normal-squad': 13,
       'crashtpp': 14, 'normal-solo': 15 }
train_df['matchType'] = train_df['matchType'].replace(matchTyp)
test_df['matchType'] = test_df['matchType'].replace(matchTyp)

train_df.dropna(inplace = True)
train_df.isnull().any().any()
X = train_df.drop(["Id", "groupId", "matchId", "winPlacePerc"], axis = 1)
y = train_df["winPlacePerc"]
test = test_df.drop(["Id", "groupId", "matchId"], axis = 1)
model = RandomForestRegressor()
model.fit(X,y)
preds = model.predict(test)

test_id = test_df["Id"]
submit_xg = pd.DataFrame({'Id': test_id, "winPlacePerc": preds} , columns=['Id', 'winPlacePerc'])
print(submit_xg.head())
submit_xg.to_csv("submission.csv", index = False)
