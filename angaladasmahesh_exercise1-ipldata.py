# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading input files



del_data = pd.read_csv('../input/ipldata/deliveries.csv')

mat_data = pd.read_csv('../input/ipldata/matches.csv')
# information about deliveries.csv



print(del_data.info())
# Checking sample data in deliveries.csv



del_data.head()
# Information about matches.csv



print(mat_data.info())
# Checking sample data in matches.csv



mat_data.head()
# number of seasons covered in given dataset in sorted order



seasons = sorted(mat_data['season'].unique())

print(seasons)
# No. of matches in each season



#    dfs = [x for _, x in mat_data.groupby('season')] #code for groupby season



for i in seasons:

#    d = (mat_data['team1'].append(mat_data['team2'])).unique()

    matches_inseason = mat_data['season'].value_counts()

print(matches_inseason)

empty_df.plot(x="Season", y=["Team Count", "Matches per Season"], kind="bar")
# Plotting number of matches and number of teams per season on same chart.



subset_raw = pd.DataFrame(mat_data, columns=['season', 'team1', 'team2'])

seasons = sorted(subset_raw['season'].unique())

empty_df = pd.DataFrame(columns=['Season', 'Team Count', 'Matches per Season', 'Teams'])



for season in seasons:

    subset = subset_raw[subset_raw['season'] == season]

    team_names = (subset['team1'].append(subset['team2'])).unique()

    matches_in_season = len(subset)

    empty_df = empty_df.append({'Season': season, 'Team Count': len(team_names),

                                'Matches per Season': matches_in_season, 'Teams': team_names}, ignore_index=True)



x = np.arange(len(seasons))

width = 0.35



ax1 = plt.subplot()

plt.xticks(x + width / 2, empty_df['Season'], rotation=90)  # will label the bars on x axis with the respective season names.

match_count = ax1.bar(x,

                      empty_df['Matches per Season'],

                      width=width,

                      label='No. of matches')

ax2 = ax1.twinx()

team_count = ax2.bar(x + width,

                     empty_df['Team Count'],

                     width=width,

                     label='No. of teams',

                     color=[.1, 1, .7, .3])

ax1.set_xlabel('Year')

ax1.set_ylabel('Number of matches(2008-2019')

ax2.set_ylabel('Number of teams')

plt.title('Total matches & Teams by season')

plt.legend([match_count, team_count],

           ['No. of matches', 'No. of teams'])

plt.show()
# Main idea is to check relation between

# power play performance and winning chance of a match



# below is runs scored in first 5 overs and last 5 over of each match separately.



play = pd.DataFrame(del_data, columns=['match_id', 'batting_team', 'over', 'total_runs'])

play_one_df = play[(play.over >= 1) & (play.over <= 5)] # powerplay one

play_four_df = play[(play.over >= 15) & (del_data.over <= 20)] # powerplay 4

team_names = play['batting_team'].unique()



# Poweplay 1;  .reset_index() kept the ouput as dataframe otherwise result is a series

test = play_one_df.groupby(['match_id', 'batting_team'])['total_runs'].sum().reset_index()



# reading match data to append win/lose status to deliveries data

match_data_slice = pd.DataFrame(mat_data,

                                columns=['id', 'winner'])



result = test.merge(match_data_slice,

                    left_on='match_id',

                    right_on='id',

                    how='left')

result['Result'] = np.where(result['batting_team'] == result['winner'],

                            'won', 'lost')

result['serial'] = result.index

pivot = result.groupby(['serial', 'Result'])['total_runs'].sum().unstack(fill_value=0)



# result1 = result[:20] - sample data for testing

df = pd.DataFrame(result, columns=['serial', 'total_runs', 'Result'])



sns.set(style="ticks")

sns.scatterplot(df['serial'], df['total_runs'], hue=df['Result'])

sns.despine() # : to remove axes

plt.xlabel('Match number -->')

plt.ylabel('Runs in 0-5 overs')

plt.title('Powerplay runs vs Match result')

plt.legend(frameon=True, loc='upper center', ncol=3)

plt.show()
