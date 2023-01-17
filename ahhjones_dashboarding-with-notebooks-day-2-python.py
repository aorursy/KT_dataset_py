from pathlib import Path  # Easy-to-use, cross-platform path-to-file.



import numpy as np

import pandas as pd



# For interactive visualisations.

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot  # Allow plots to display inside notebook.

init_notebook_mode()
path_to_data = Path('../input')



# Load users data.

users = pd.read_csv(path_to_data / 'Users.csv', parse_dates=['RegisterDate'], dayfirst=False)



# Load competitions data (takes a little while).

competitions = pd.read_csv(path_to_data / 'Competitions.csv', 

                           parse_dates=['EnabledDate', 'DeadlineDate', 'ProhibitNewEntrantsDeadlineDate'], 

                           dayfirst=False)
# Group by RegisterDate and count unique Id values.

new_users_per_day = users.groupby('RegisterDate').agg({'Id': 'nunique'}).rename({'Id': 'NewUsers'}, axis=1)

new_users_per_day = new_users_per_day.resample('1W').sum()



# Specify plot.

data = [go.Scatter(x=new_users_per_day.index, y=new_users_per_day['NewUsers'], text=new_users_per_day.index,

                  hoverinfo='y+text')]



layout = {

    'title': "Number of new Kaggle users per week",

    'xaxis': {

        'title': 'Date',

        'zeroline': False

    },

    'yaxis': {

        'title': 'Number of new users'

    }

}



# Create and display plot.

fig = {'data': data, 'layout': layout}

iplot(fig)
# Replace USD & EUR --> Cash.

competitions.replace(['USD', 'EUR'], 'Cash', inplace=True)



# Median number of competitors for each reward type.

agg_dict = {

    'TotalCompetitors': ['median', 'min', 'max', 'mean'], 

    'Id': 'nunique'

}

rewardtype_vs_competitors = competitions.groupby('RewardType').agg(agg_dict)

rewardtype_vs_competitors.sort_values(by=('TotalCompetitors', 'median'), ascending=False, inplace=True)



# Specify plot.

text = rewardtype_vs_competitors[('TotalCompetitors', 'median')].apply(lambda x: 'Median: %.0f <br>' % x)

text += rewardtype_vs_competitors[('TotalCompetitors', 'mean')].apply(lambda x: 'Mean: %.0f <br>' % x)

text += rewardtype_vs_competitors[('TotalCompetitors', 'min')].apply(lambda x: 'Minimum: %s <br>' % x)

text += rewardtype_vs_competitors[('TotalCompetitors', 'max')].apply(lambda x: 'Maximum: %s <br>' % x)

text += rewardtype_vs_competitors[('Id', 'nunique')].apply(lambda x: '# competitions: %s' % x)

data = [go.Bar(x=rewardtype_vs_competitors.index, 

               y=rewardtype_vs_competitors[('TotalCompetitors', 'median')], 

               text=text,

               hoverinfo='text')]



layout = {

    'title': "Median number of entrants per competition reward type",

    'xaxis': {

        'title': 'Reward type', 

        'zeroline': False

    },

    'yaxis': {

        'title': 'Median competitors'

    }

}



# Create and display plot.

fig = {'data': data, 'layout': layout}

iplot(fig)