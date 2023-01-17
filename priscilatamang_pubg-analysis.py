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
import matplotlib.pyplot as plt
import seaborn as sns
pubg = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/train_V2.csv")
#Data Wrangling
pubg.head(10)
totalGames = pubg.shape[0]
pubg.info()
pubg.isnull().sum()

pubg.describe()
pubg.sample(5)
pubg[pubg["Id"].duplicated()]
#Findings:
#One null value in winPlacePerc.
#Huge inconsistency of data (nothing can be done).
## Dealing with null value in winPlacePerc
pubg["winPlacePerc"].fillna(pubg["winPlacePerc"].mean(), inplace=True)
pubg.info()

pubg.isnull().sum()
#Exploratory Data Analysis
pubg.columns
# pubg["killPlace"] = pubg["killPlace"].astype("category")
pubg["matchType"] = pubg["matchType"].astype("category")
pubg["maxPlace"] = pubg["maxPlace"].astype("category")
pubg["numGroups"] = pubg["numGroups"].astype("category")
pubg["assists"] = pubg["assists"].astype("category")
pubg["headshotKills"] = pubg["headshotKills"].astype("category")
pubg["revives"] = pubg["revives"].astype("category")
pubg["roadKills"] = pubg["roadKills"].astype("category")
pubg["teamKills"] = pubg["teamKills"].astype("category")
pubg["vehicleDestroys"] = pubg["vehicleDestroys"].astype("category")
pubg["DBNOs"] = pubg["DBNOs"].astype("category")
pubg["boosts"] = pubg["boosts"].astype("category")
pubg["heals"] = pubg["heals"].astype("category")
pubg["kills"] = pubg["kills"].astype("category")
pubg["killStreaks"] = pubg["killStreaks"].astype("category")
pubg.info()
#Univariate Analysis On All Columns
columnsToBeAnalyzed = pubg.columns[3:]
for i in columnsToBeAnalyzed:
    print(i)
    if(pubg[i].dtype.name == "category"):
        print((pubg[i].value_counts()/totalGames)*100)
        sns.countplot(pubg[i])
    else:
        try:
            sns.distplot(pubg[i])
        except:
            sns.distplot(pubg[i], kde=False)
        print("Skew =", pubg[i].skew())
        print("Kurtosis =", pubg[i].kurt())
    plt.show()