# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(sorted(os.listdir("../input")))

# Any results you write to the current directory are saved as output.
teams = pd.read_csv('../input/Teams.csv').rename(columns={'Id':'TeamId'})
teams.head()
team_memberships = pd.read_csv('../input/TeamMemberships.csv')
team_memberships.head()
users = pd.read_csv('../input/Users.csv').rename(columns={'Id':'UserId'})
users.head()
tmp_df = team_memberships.groupby('TeamId').UserId.count().to_frame('Size').reset_index()
teams = teams.merge(tmp_df, how='left', on='TeamId')
teams.head()  
teams.Medal.value_counts()
solo_teams = teams[(teams.Size == 1) & (teams.Medal == 1)]
solo_teams = solo_teams.merge(team_memberships, how='left', on='TeamId')
solo_teams = solo_teams.merge(users, how='left', on='UserId')
solo_teams.head()
solo_gold = solo_teams.groupby(['UserName', 'DisplayName']).Medal.count().sort_values(ascending=False)
solo_gold.head(20)
solo_teams['LastSubDate'] = pd.to_datetime(solo_teams.LastSubmissionDate)
recent_solo_teams = solo_teams[solo_teams.LastSubDate > '2016-07-11']
recent_solo_gold = recent_solo_teams.groupby(['UserName', 'DisplayName']).Medal.count().sort_values(ascending=False)
recent_solo_gold.head(20)