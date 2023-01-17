# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Libraries for Plotting

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('fivethirtyeight')

#Libraries used for Machine Learning (Predictions)

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Load all the data as Pandas Dataframe

df_seeds = pd.read_csv('../input/WNCAATourneySeeds.csv')
df_tour = pd.read_csv('../input/WNCAATourneyCompactResults.csv')
df_teams = pd.read_csv('../input/WTeams.csv')
df_cities = pd.read_csv('../input/WCities.csv')
df_gamecities = pd.read_csv('../input/WGameCities.csv')
df_tourneyslots = pd.read_csv('../input/WNCAATourneySlots.csv')
df_regseasoncompactresults = pd.read_csv('../input/WRegularSeasonCompactResults.csv')
df_seasons = pd.read_csv('../input/WSeasons.csv')
df_teamspellings = pd.read_csv('../input/WTeamSpellings.csv', engine='python')
print(df_tour.shape)
print(df_regseasoncompactresults.shape)
df_seeds.head()
# Convert Tourney Seed to a Number
df_seeds['SeedNumber'] = df_seeds['Seed'].apply(lambda x: int(x[-2:]))
df_seeds.head()
# Merging tables

df_gamecities = df_gamecities.merge(df_cities,how='left',on='CityID')
df_gamecities.head()
df_tour['WSeed'] = df_tour[['Season','WTeamID']].merge(df_seeds,left_on = ['Season','WTeamID'],right_on = ['Season','TeamID'],how='left')[['SeedNumber']]
df_tour.head()
# Calculate the Average Team Seed
df_average_seed = df_seeds.groupby(['TeamID']).agg(np.mean).sort_values('SeedNumber')
df_average_seed = df_average_seed.merge(df_teams, left_index=True, right_on='TeamID') #Add Teamnname
#Plotting Top 25 Average Tournament Seed

df_average_seed.head(25).sort_values('SeedNumber').plot(x='TeamName',
                              y='SeedNumber',
                              kind='bar',
                              figsize=(15,5),
                              title='Top 25 Average Tournament Seed'
                             )
df_tour.head()
df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tour.head()
