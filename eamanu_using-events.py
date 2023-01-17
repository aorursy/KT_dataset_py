# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
events_2017 = pd.read_csv('../input/Events_2017.csv')
print (events_2017.columns)
events_winner = events_2017.query('WTeamID == EventTeamID').drop(['EventID', 'LTeamID','EventTeamID',
                                                                  'LPoints','ElapsedSeconds'], axis=1)
events_losser = events_2017.query('LTeamID == EventTeamID').drop(['EventID', 'WTeamID','EventTeamID',
                                                                  'WPoints', 'ElapsedSeconds'], axis=1)

event_dummies_winner = pd.get_dummies(events_winner).groupby(['Season', 'DayNum', 'WTeamID'], as_index=False)
event_dummies_losser = pd.get_dummies(events_losser).groupby(['Season', 'DayNum', 'LTeamID'], as_index=False)

# Winner data frame
e2017_count_winner = event_dummies_winner['EventType_assist'].sum().rename(columns={'EventType_assist':'Wassist'})
e2017_count_winner['Wblock'] = event_dummies_winner['EventType_block'].sum()['EventType_block']
e2017_count_winner['Wsteal'] = event_dummies_winner['EventType_steal'].sum()['EventType_steal']
e2017_count_winner['Wturnover'] = event_dummies_winner['EventType_turnover'].sum()['EventType_turnover']
e2017_count_winner['Wreb_off'] = event_dummies_winner['EventType_reb_off'].sum()['EventType_reb_off']
e2017_count_winner['Wreb_def'] = event_dummies_winner['EventType_reb_def'].sum()['EventType_reb_def']
# Losser Data frame
e2017_count_losser = event_dummies_losser['EventType_assist'].sum().rename(columns={'EventType_assist':'Lassist'})
e2017_count_losser['Lblock'] = event_dummies_losser['EventType_block'].sum()['EventType_block']
e2017_count_losser['Lsteal'] = event_dummies_losser['EventType_steal'].sum()['EventType_steal']
e2017_count_losser['Lturnover'] = event_dummies_losser['EventType_turnover'].sum()['EventType_turnover']
e2017_count_losser['Lreb_off'] = event_dummies_losser['EventType_reb_off'].sum()['EventType_reb_off']
e2017_count_losser['Lreb_def'] = event_dummies_losser['EventType_reb_def'].sum()['EventType_reb_def']
f = (
    e2017_count_winner.loc[:,['Wassist', 'Wblock','Wsteal',
                               'Wturnover', 'Wreb_off', 'Wreb_def']]
    .dropna()
    ).corr()

sns.heatmap(f, annot=True)

e2017 = pd.concat([e2017_count_winner, e2017_count_losser.drop(['Season', 'DayNum'], axis=1)], axis = 1)
e2017.sample(50).sort_index().plot(x='DayNum', y=['Wassist', 'Lassist'])