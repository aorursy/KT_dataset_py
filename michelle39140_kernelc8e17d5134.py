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
# load model
from sklearn.externals import joblib
rf_test = joblib.load('../input/pretrainedpubgmodel/rf_team_simple.joblib') 
test_df=pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")
test_groupData=pd.DataFrame(test_df.groupby('groupId')[['assists', 'boosts', 'damageDealt', 'DBNOs','headshotKills', 'heals', 'killPlace', 'killPoints', 'kills','killStreaks', 'longestKill', 'matchDuration', 'maxPlace','numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills','swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance','weaponsAcquired', 'winPoints']].mean())
test_combinedData=test_groupData.join(test_groupData,rsuffix="_team",on="groupId")
test_combinedData.dropna(inplace=True)
test_combinedData.head()
test_Data = test_combinedData[['boosts_team','damageDealt_team','DBNOs_team','killPlace_team','kills_team','killStreaks_team','longestKill_team','walkDistance_team','weaponsAcquired_team']]
test_Data.head()
predictions = rf_test.predict(test_Data)
test_Data["results"]=predictions

group_result=test_Data[["results"]]
id_n_group=test_df[["Id","groupId"]]

final_result=id_n_group.join(group_result,on="groupId")
final_result.drop(["groupId"],inplace=True,axis=1)

final_result.head()
final_result.rename(columns={"results":"winPlacePerc"},inplace=True)
final_result.set_index("Id",inplace = True)
final_result.head()
final_result.to_csv("prediction_result.csv")