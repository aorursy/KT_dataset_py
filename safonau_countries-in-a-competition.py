# Parsing contry and other data that does not exist in the Meta Kaggle dataset

import requests

from bs4 import BeautifulSoup

import time

import re

import json
import pandas as pd

from pathlib import Path



INPUT_FOLDER = Path('/kaggle/input/meta-kaggle/')
REQUEST_DELAY = 0.5  # for time.sleep
competitions = pd.read_csv(INPUT_FOLDER / 'Competitions.csv')

teams = pd.read_csv(INPUT_FOLDER / 'Teams.csv')

users = pd.read_csv(INPUT_FOLDER / 'Users.csv' )

team_memberships = pd.read_csv(INPUT_FOLDER / 'TeamMemberships.csv')
display(competitions[competitions.HostSegmentTitle == 'Featured'].head(3))
# A value from the Slug column

COMPETITION_SLUG = '3d-object-detection-for-autonomous-vehicles'
competition_id = competitions.loc[competitions.Slug == COMPETITION_SLUG, 'Id'].values[0]

competition_id
competition_team = teams[(teams.CompetitionId == competition_id) & (~teams.PublicLeaderboardRank.isnull())]
len(competition_team)
competition_team.head()
competition_participants = (

    competition_team

    .merge(team_memberships, left_on='Id', right_on='TeamId', how='left')

    .merge(users, left_on='UserId', right_on='Id', how='left')[

        ['TeamName', 'PublicLeaderboardRank', 'PrivateLeaderboardRank', 'Medal', 'UserName']]

).sort_values(by='PublicLeaderboardRank')
competition_participants.head(10)
KAGGLE_BASE_URL = "https://kaggle.com/"
usernames = competition_participants['UserName'].dropna()
subset_fields = set(['city', 'region', 'country', 'occupation', 'organization'])

users_data = []

for username in usernames.head(50):

    time.sleep(REQUEST_DELAY)

    profile_url = f'{KAGGLE_BASE_URL}{username}'

    

    result = requests.get(profile_url)

    src = result.text

    soup = BeautifulSoup(src, 'html.parser').find_all("div", id="site-body")[0].find("script")

    

    user_info = re.search('Kaggle.State.push\(({.*})', str(soup)).group(1)

    user_dict = json.loads(user_info)



    user_subset = {k:v  for k, v in user_dict.items() if k in subset_fields}

    user_subset.update({'username': username})



    users_data.append(user_subset)
users_data_df = pd.DataFrame(users_data)

users_data_df.head(10)
country_counts = users_data_df.country.value_counts(dropna=False)

country_counts
country_counts.plot(kind='bar');