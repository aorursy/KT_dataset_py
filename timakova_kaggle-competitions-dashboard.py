# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import datetime

import plotly

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.io as pio

init_notebook_mode(connected=True)

competitions = pd.read_csv('../input/Competitions.csv', low_memory=False)

#datasets = pd.read_csv('../input/Datasets.csv')

teams = pd.read_csv('../input/Teams.csv', low_memory=False)

teammemberships = pd.read_csv('../input/TeamMemberships.csv', low_memory=False)



## dates seem to be recorded as "objects" while we would prefer datetime. Let's convert!

competitions['DeadlineDate'] = pd.to_datetime(competitions['DeadlineDate'], infer_datetime_format=True)

# number of competitions

competitions_count = competitions.groupby(competitions['DeadlineDate'].dt.year).count()

competitions_sum = competitions.groupby(competitions['DeadlineDate'].dt.year).sum()

## Remove all rows with NA and also rows with non-dollar reward prize (we could keep and convert EUR, but for now let's disregard)

prize = competitions.dropna(how='all')

prize = prize.loc[prize["RewardType"] == "USD"]

prize = prize.groupby(prize['DeadlineDate'].dt.year).median()

h_organizations = pd.DataFrame(data = competitions[['OrganizationId', 'DeadlineDate']], 

                               columns=['OrganizationId', 'DeadlineDate']).dropna()

h_organizations = h_organizations[h_organizations.OrganizationId != 4.0] ## exclude Kaggle

un_h_orgs = h_organizations.groupby(h_organizations['DeadlineDate'].dt.year)['OrganizationId'].nunique() ## unique per year

competitors_count = competitions.groupby(competitions['DeadlineDate'].dt.year)['TotalCompetitors'].sum()
x1 = competitions_count.DeadlineDate.index

y1 = competitions_count.Id

x2 = x1

y2 = un_h_orgs

x3 = x1

y3 = competitors_count

x4 = x1

y4 = prize.RewardQuantity





trace1 = go.Scatter(

    x = x1,

    y = y1,

    mode = 'lines+markers',

    name = 'competitions'

)

trace2 = go.Scatter(

    x = x2,

    y = y2,

    mode = 'lines+markers',

    name = 'organizations'

)



trace3 = go.Scatter(

    x = x3,

    y = y3,

    mode = 'lines+markers',

    name = 'contestants'

)

trace4 = go.Scatter(

    x = x4,

    y = y4,

    mode = 'lines+markers',

    name = 'Dollars'

)

fig = plotly.subplots.make_subplots(rows=2, cols=2, print_grid=False, subplot_titles=('Competitions', 'Hosting Organizations (exclud. Kaggle)',

                                                          'Contestants', 'Median Prize'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 2, 1)

fig.append_trace(trace4, 2, 2)



fig['layout']['xaxis1'].update(title='Year')

fig['layout']['xaxis2'].update(title='Year')

fig['layout']['xaxis3'].update(title='Year')

fig['layout']['xaxis4'].update(title='Year')



fig['layout']['yaxis1'].update(title='# of Competitions')

fig['layout']['yaxis2'].update(title='# of Organizations')

fig['layout']['yaxis3'].update(title='# of Contestants')

fig['layout']['yaxis4'].update(title='# of Submissions')



fig['layout'].update(showlegend=False, title='Basic Metrics of Kaggle\'s success')



iplot(fig)



## let's add some more columns to our df, just in case

overview = pd.DataFrame(data=competitions, 

                        columns=["Id", "Title", "DeadlineDate", "LeaderboardPercentage", 

                                 "TotalSubmissions", "NumScoredSubmissions", "TotalCompetitors", 

                                 "TotalTeams", "RewardType", "RewardQuantity"])

## Remove all rows with NA and also rows with non-dollar reward prize 

##(we could keep and convert EUR, but for now let's disregard)

overview = overview.dropna(how='all')

overview = overview.loc[overview["RewardType"] == "USD"]



N = overview.shape[0]

x = overview["DeadlineDate"]

y = overview["RewardQuantity"]

comp_name = overview["Title"]

colors = np.random.rand(N)

sz = overview["TotalTeams"]

layout = go.Layout(

                        hovermode="closest", 

                        title= "Kaggle $-prize Competitions. Size of a bubble reflects number of teams. Zoom!",

                        xaxis=dict(

                            title='Deadline Year'),

                        yaxis=dict(

                            title='Prize (USD)')

                    )

fig = go.Figure(layout=layout)

fig.add_scatter(x=x,

                y=y,

                mode='markers',

                hoverinfo='text+y',

                hovertext=comp_name,

                hoveron='points+fills',

                marker={'size': sz,

                        'sizeref': 50,

                        'sizemin': 3,

                        'color': colors,

                        'opacity': 0.6,

                        'colorscale': 'Viridis',

                        "line": {

                            "width": 0

                            }

                       },

                    );



iplot(fig)
## create a new df with number of people in each team (use teammemberships)

headcount = teammemberships.groupby(teammemberships['TeamId']).count()

## join headcount and teams on teamId, get rid of irrelevant columns

two_tables = pd.merge(teams, headcount, how='right', left_on="Id", right_on="TeamId",sort=False, copy=True)

two_tables = two_tables.rename(index=str, columns={"UserId": "Headcount"})

two_tables = two_tables.drop(columns=['Id_x', 'Id_y', 'RequestDate'])

teams_contests = pd.merge(two_tables, overview, how='right', left_on="CompetitionId", right_on="Id",

                          sort=False, copy=True)

teams_contests = teams_contests.dropna(subset=['PublicLeaderboardRank', 'Headcount'])



#create 4 buckets for team size: 1, 2-4, 5-12, >12. Plot distribution of the rating for all 4

solo = teams_contests[teams_contests.Headcount == 1]

small_teams = teams_contests[(teams_contests.Headcount >= 2) & (teams_contests.Headcount <= 4)]

medium_teams = teams_contests[(teams_contests.Headcount >= 5) & (teams_contests.Headcount <= 10)]

big_teams = teams_contests[teams_contests.Headcount >= 11]



x1 = np.log(solo["PublicLeaderboardRank"]) 

x2 = np.log(small_teams["PublicLeaderboardRank"]) 

x3 = np.log(medium_teams["PublicLeaderboardRank"])

x4 = np.log(big_teams["PublicLeaderboardRank"]) 



hist_data = [x1, x2, x3, x4]



group_labels = ['1', '2-4', '5-10', '11+']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

fig['layout'].update(title='Leaderboard Rank as a Function of Team Size',

                     yaxis=dict(title='Probability'),

                     xaxis=dict(title='Log of Rank (smaller = better)'),

                     annotations=[

                                    dict(

                                        x=1.03,

                                        y=1.05,

                                        align="right",

                                        valign="top",

                                        text='# of People in a Team',

                                        showarrow=False,

                                        xref="paper",

                                        yref="paper",

                                        xanchor="center",

                                        yanchor="top"

                                    )

                                ]

                    )

iplot(fig)