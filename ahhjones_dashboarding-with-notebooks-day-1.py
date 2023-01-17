# Download and unzip data.
from pathlib import Path  # Easy-to-use, cross-platform path-to-file.



import numpy as np

import pandas as pd
path_to_data = Path('../input')



# Load users data.

users = pd.read_csv(path_to_data / 'Users.csv', parse_dates=['RegisterDate'], dayfirst=False)

users.head()
# Load competitions data (takes a little while).

competitions = pd.read_csv(path_to_data / 'Competitions.csv', 

                           parse_dates=['EnabledDate', 'DeadlineDate', 'ProhibitNewEntrantsDeadlineDate'], 

                           dayfirst=False)

competitions.head()
import matplotlib.pyplot as plt  # Quick plotting.

import seaborn  # Make plotting better.

%matplotlib inline
# Group by RegisterDate and count unique Id values.

new_users_per_day = users.groupby('RegisterDate').agg({'Id': 'nunique'}).rename({'Id': 'NewUsers'}, axis=1)



new_users_per_day.plot(title='New users per day', figsize=(15, 7), grid=True, legend=False)
# Replace USD & EUR --> Cash.

competitions.replace(['USD', 'EUR'], 'Cash', inplace=True)



# Median number of competitors for each reward type.

rewardtype_vs_competitors = competitions.groupby('RewardType').agg({'TotalCompetitors': 'median'})

rewardtype_vs_competitors.sort_values(by='TotalCompetitors', ascending=False, inplace=True)



ax = rewardtype_vs_competitors.plot(title='Competitors by reward type', 

                                    figsize=(15, 7), grid=True, legend=False, kind='bar')

ax.set_xlabel('Reward type')

ax.set_ylabel('Median number of competitors')