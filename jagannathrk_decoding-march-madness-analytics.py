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
# Importing the libraries

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

from IPython.display import display


Wteams = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WTeams.csv')

Wteams.head()
# No of Teams



Wteams['TeamID'].nunique()
Wseason = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WSeasons.csv')

Wseason.tail()
# Total held seasons including the current

Wseason['Season'].count()
Wseeds = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WNCAATourneySeeds.csv')

Wseeds.head()
Wseeds = pd.merge(Wseeds, Wteams,on='TeamID')

Wseeds.head()

# Separating the regions from the Seeds



Wseeds['Region'] = Wseeds['Seed'].apply(lambda x: x[0][:1])

Wseeds['Seed'] = Wseeds['Seed'].apply(lambda x: int(x[1:3]))

print(Wseeds.head())

print(Wseeds.shape)
# Teams with maximum top seeds



colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow'] 



Wseeds[Wseeds['Seed'] ==1]['TeamName'].value_counts()[:10].plot(kind='bar',color=colors,linewidth=2,edgecolor='black')

plt.xlabel('Number of times in Top seeded positions')
# Teams with maximum lowest seeds

Wseeds[Wseeds['Seed'] ==16]['TeamName'].value_counts()[:10].plot(kind='bar',color=colors,edgecolor='black',linewidth=1)

plt.xlabel('Number of times in bottom seeded positions')
rg_season_compact_results = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

rg_season_compact_results.head()
# Winning and Losing score Average over the years

x = rg_season_compact_results.groupby('Season')[['WScore','LScore']].mean()



fig = plt.gcf()

fig.set_size_inches(14, 6)

plt.plot(x.index,x['WScore'],marker='o', markerfacecolor='green', markersize=12, color='green', linewidth=4)

plt.plot(x.index,x['LScore'],marker=7, markerfacecolor='red', markersize=12, color='red', linewidth=4)

plt.legend()

tourney_compact_results = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')

tourney_compact_results .tail()
games_played = tourney_compact_results.groupby('Season')['DayNum'].count().to_frame().merge(rg_season_compact_results.groupby('Season')['DayNum'].count().to_frame(),on='Season')

games_played.rename(columns={"DayNum_x": "Tournament Games", "DayNum_y": "Regular season games"})
