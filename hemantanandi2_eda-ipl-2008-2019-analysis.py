# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
inp0 = pd.read_csv('../input/ipldata/deliveries.csv')

inp1 = pd.read_csv('../input/ipldata/matches.csv')
first_inning = inp0[inp0.inning==1]

first_inning_batting = first_inning.groupby(by=['batting_team'])['total_runs'].sum()

first_inning_total_match = first_inning.groupby(by=['batting_team'])['match_id'].nunique()

first_inning_team_average = first_inning_batting/first_inning_total_match

first_inning_team_average = first_inning_team_average.reset_index()

first_inning_team_average['inning'] = 1
second_inning = inp0[inp0.inning==2]

second_inning_batting = second_inning.groupby(by=['batting_team'])['total_runs'].sum()

second_inning_total_match = second_inning.groupby(by=['batting_team'])['match_id'].nunique()

second_inning_team_average = second_inning_batting/second_inning_total_match

second_inning_team_average = second_inning_team_average.reset_index()

second_inning_team_average['inning'] = 2
team_average = pd.concat([first_inning_team_average,second_inning_team_average])

team_average.rename(columns={0: "average_run"},inplace=True)
###Average runs scored by each team throughout the seasons

plt.figure(figsize=(16,8))

ax=sns.barplot(data=team_average.sort_values(by='average_run',ascending=False),hue='inning',x='batting_team',y='average_run')

for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),

                    ha='right', va='top', color= 'white',fontweight='bold',fontsize=14,rotation=90)

plt.xlabel("Name of the team",FontSize=12)

plt.ylabel("Average runs",FontSize=12)

plt.xticks(rotation=90,fontsize=16)

plt.title('Average runs scored by each team during IPL 2008 - 2019 (First Batting vs Second Batting)',fontsize=18)

plt.show()