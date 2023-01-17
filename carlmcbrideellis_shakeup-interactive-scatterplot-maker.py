import numpy as np

import pandas as pd

import plotly.express as px

import warnings

warnings.filterwarnings('ignore')
search_string = "OSIC"
comps = pd.read_csv('../input/meta-kaggle/Competitions.csv')

our_competition  = comps[comps['Title'].str.contains(search_string,na=False)]

pd.set_option('display.max_columns', None)

our_competition
CompetitionId = our_competition["Id"].squeeze()

CompetitionIndex = our_competition.index.values.astype(int)[0]

all_teams = pd.read_csv('../input/meta-kaggle/Teams.csv')

teams = all_teams[all_teams['CompetitionId']==CompetitionId]

teams = teams.assign(Medal=teams.Medal.fillna(0).astype(int))

COLOR_DICT = {0: 'deepskyblue', 1: 'gold', 2: 'silver', 3: 'chocolate'}

MEDAL_NAMES = np.asarray(["None", "Gold", "Silver", "Bronze"])

MEDAL_COLORS = dict(zip(MEDAL_NAMES, COLOR_DICT.values()))

row = comps.loc[CompetitionIndex]

teams = teams.assign(Medal=MEDAL_NAMES[teams.Medal])

fig = px.scatter(teams,

                 title='Shakeup plot for: ' + row.Title,

                 x='PublicLeaderboardRank',

                 y='PrivateLeaderboardRank',

                 hover_name='TeamName',

                 hover_data=[

                     'ScoreFirstSubmittedDate',

                     'LastSubmissionDate',

                     'PublicLeaderboardSubmissionId',

                     'PrivateLeaderboardSubmissionId',

                     'Medal',

                     'MedalAwardDate',

                 ],

                 color='Medal',

                 color_discrete_map=MEDAL_COLORS)

fig.update_traces(marker=dict(size=5))

fig.update_layout(showlegend=False)

fig.show()