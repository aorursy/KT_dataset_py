# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Importing useful packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
# Input data files are available in the "../input/" directory.

# List the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Use function .read_csv to read CSV files into DataFrame



# Read game results since 1985

df_compact = pd.read_csv('../input/datafiles/RegularSeasonCompactResults.csv')

df_compact.head(10)
# Read team info

df_teams = pd.read_csv('../input/datafiles/Teams.csv')

df_teams['TeamName'].head()
# Read play by play data for year 2018

play2018 = pd.read_csv('../input/playbyplay_2018/Events_2018.csv')

play2018.head()
# Read tournament seed data

df_seeds = pd.read_csv('../input/datafiles/NCAATourneySeeds.csv')

df_seeds.head()
# Check the data size for the dataframe for teams

df_teams.shape
# Check the data type for each columns in teams data

df_teams.dtypes
# Count of unique values of each columns in the teams data. One can see that there are 366 unique teams

df_teams.nunique()
df_pbp = pd.DataFrame()

for i in range(8):

    df = pd.read_csv('../input/playbyplay_201' + str(i) + '/Events_201' + str(i) + '.csv')

    df_pbp = df_pbp.append(df)

    print("Cumulative data size for year 201" + str(i) + ": " + str(df_pbp.shape))
df_seeds['seed_int'] = df_seeds['Seed'].apply(lambda x: int(x[1:3]))

df_winseeds = df_seeds.loc[:, ['TeamID', 'Season', 'seed_int']].rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})

df_lossseeds = df_seeds.loc[:, ['TeamID', 'Season', 'seed_int']].rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})

df_temp = pd.merge(left=df_compact, right=df_winseeds, how='left', on=['Season', 'WTeamID'])

df_concat = pd.merge(left=df_temp, right=df_lossseeds, on=['Season', 'LTeamID'])



df_concat.head()
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed

df_concat.head()
df_wins = pd.DataFrame()

df_wins['SeedDiff'] = df_concat['SeedDiff']

df_wins['Result'] = 1



df_losses = pd.DataFrame()

df_losses['SeedDiff'] = -df_concat['SeedDiff']

df_losses['Result'] = 0



df_predictions = pd.concat((df_wins, df_losses))

df_predictions.head()
X_train = df_predictions.SeedDiff.values.reshape(-1,1)

y_train = df_predictions.Result.values

X_train, y_train = shuffle(X_train, y_train)
X_train.shape
y_train.shape