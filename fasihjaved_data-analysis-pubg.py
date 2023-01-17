# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv('../input/train.csv')
df.info()

# Any results you write to the current directory are saved as output.
# Isolate the solo games
solo_df = pd.DataFrame()
solo_df = df[df["numGroups"]>50].copy()
solo_df.info()
# Checking if there aren't any duplicates or na's
solo_df.duplicated().value_counts()
solo_df.isna().any()
solo_df["revives"].value_counts()
solo_df["DBNOs"].value_counts()
solo_df.drop(["Id", "groupId", "DBNOs", "teamKills", "winPoints", "numGroups", "revives", "killPoints"], axis=1, inplace=True)
solo_df["kills"].describe()
solo_df[solo_df["kills"]==42]
solo_df[(solo_df["matchId"]==1441)&(solo_df["winPlacePerc"]>0.9)]
solo_df["killStreaks"].describe()

solo_df[solo_df["killStreaks"]==10]
solo_df[(solo_df["matchId"]==45605)&(solo_df["winPlacePerc"]>0.9)]
solo_df[(solo_df["matchId"]==21204)&(solo_df["winPlacePerc"]>0.9)]
solo_df[(solo_df["kills"]==0)&(solo_df["winPlacePerc"]==1.0)]
solo_df[solo_df["matchId"]==13392].describe()
solo_df[(solo_df["matchId"]==13392)&(solo_df["winPlacePerc"]==1.0)]
solo_df["kills"][solo_df["winPlacePerc"]==1.0].describe()
solo_df["kills"][solo_df["winPlacePerc"]<1.0].describe()
solo_df["kills"][(solo_df["winPlacePerc"]<1.0)&(solo_df["kills"]>0.0)].describe()
solo_df["kills"][(solo_df["winPlacePerc"]>=0.90)&(solo_df["winPlacePerc"]<1.0)&(solo_df["kills"]>0.0)].describe()
solo_df["kills"].value_counts().head(10)
solo_df["kills"][(solo_df["winPlacePerc"]<=0.25)].value_counts()
solo_df["weaponsAcquired"][(solo_df["winPlacePerc"]<=0.25)].value_counts()
solo_df["longestKill"][(solo_df["winPlacePerc"]<=0.25)].mean()
solo_df["longestKill"][(solo_df["winPlacePerc"]>0.25)&(solo_df["winPlacePerc"]<=0.50)].mean()
plt.plot(solo_df["walkDistance"][(solo_df["winPlacePerc"]<=0.25)&(solo_df["walkDistance"]>0.0)].value_counts().sort_index().index.values,
        solo_df["walkDistance"][(solo_df["winPlacePerc"]<=0.25)&(solo_df["walkDistance"]>0.0)].value_counts().sort_index(), alpha=0.5, label="First 25%")
plt.plot(solo_df["walkDistance"][(solo_df["winPlacePerc"]>0.25)&(solo_df["winPlacePerc"]<=0.50)&(solo_df["walkDistance"]>0.0)].value_counts().sort_index().index.values,
        solo_df["walkDistance"][(solo_df["winPlacePerc"]>0.25)&(solo_df["winPlacePerc"]<=0.50)&(solo_df["walkDistance"]>0.0)].value_counts().sort_index(), alpha=0.5, label="25%-50%")
plt.legend()
plt.show()
plt.plot(solo_df["rideDistance"][(solo_df["winPlacePerc"]<=0.25)&(solo_df["rideDistance"]>0.0)].value_counts().sort_index().index.values,
        solo_df["rideDistance"][(solo_df["winPlacePerc"]<=0.25)&(solo_df["rideDistance"]>0.0)].value_counts().sort_index(), alpha=0.5, label="First 25%")
plt.plot(solo_df["rideDistance"][(solo_df["winPlacePerc"]>0.25)&(solo_df["winPlacePerc"]<=0.50)&(solo_df["rideDistance"]>0.0)].value_counts().sort_index().index.values,
        solo_df["rideDistance"][(solo_df["winPlacePerc"]>0.25)&(solo_df["winPlacePerc"]<=0.50)&(solo_df["rideDistance"]>0.0)].value_counts().sort_index(), alpha=0.5, label="25%-50%")
plt.legend()
plt.show()
solo_df["walkDistance"][(solo_df["winPlacePerc"]<=0.25)&(solo_df["rideDistance"]>0.0)].mean()


