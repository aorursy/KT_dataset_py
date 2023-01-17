# imports

from os import path

import pandas as pd

import numpy as np

import sqlite3



# plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



# config

path_root = '..'

path_data = path.join(path_root, 'input')

con = sqlite3.connect(path.join(path_data, 'database.sqlite'))
# read data from sqlite db

usd_competitions = pd.read_sql_query("""

select

    c.Id

    ,c.CompetitionName

    ,c.RewardQuantity

    ,c.DateEnabled

    ,c.Deadline

    ,c.MaxDailySubmissions

    ,count(s.id) as submission_count

from competitions c

inner join teams t

    on t.CompetitionId = c.Id

inner join submissions s

    on s.teamid = t.id

where 1=1

    and c.rewardtypeid in (select id from rewardtypes where name = 'USD')  -- filter for competitions with USD rewards

    and c.RewardQuantity > 1  -- filter for competitions with USD rewards > $1

group by

    c.Id

    ,c.CompetitionName

    ,c.RewardQuantity

    ,c.DateEnabled

    ,c.Deadline

    ,c.MaxDailySubmissions

order by submission_count desc

""", con)



print('Fetched {:,} records with {:,} columns.'.format(*usd_competitions.shape))
# create features and clean up data

usd_competitions['date_enabled'] = pd.to_datetime(usd_competitions.DateEnabled)

usd_competitions['deadline'] = pd.to_datetime(usd_competitions.Deadline)

usd_competitions['competition_year'] = usd_competitions.date_enabled.dt.year

usd_competitions['ln_submission_count'] = np.log(usd_competitions.submission_count.fillna(1))

usd_competitions['duration'] = (usd_competitions.deadline - usd_competitions.date_enabled).dt.days

usd_competitions = usd_competitions[usd_competitions.RewardQuantity > 1]

usd_competitions['ln_reward'] = np.log(usd_competitions.RewardQuantity)



# exclude competition `flight2-final` because it doesn't sound or look like a real competition

usd_competitions = usd_competitions[usd_competitions.CompetitionName != 'flight2-final']

print('Cleaned dataset has {:,} records with {:,} columns.'.format(*usd_competitions.shape))
size = usd_competitions.RewardQuantity

data = [go.Scatter(

    x = usd_competitions.ln_reward,

    y = usd_competitions.ln_submission_count,

    mode = 'markers',

    text=usd_competitions.CompetitionName,

    )]



layout = go.Layout(

    title='Competition submission count by reward',

    hovermode = 'closest',

    yaxis=dict(title='Log submission count'),

    xaxis=dict(title='Log reward (USD)'),

)



figure=go.Figure(data=data,layout=layout)



py.iplot(figure, filename='scatter-mode')
size = usd_competitions.RewardQuantity

data = [go.Scatter(

    x = usd_competitions.deadline,

    y = usd_competitions.ln_submission_count,

    mode = 'markers',

    marker=dict(

        size=size,

        sizemode='area',

        sizeref=.003 * max(size),

        sizemin=3,

        color=np.log(np.clip(usd_competitions.duration, 1, None)),

        showscale=True,

    ),

        text=usd_competitions.CompetitionName,

    )]



layout = go.Layout(

    title='Competition submission count over time (area proportional to reward, color indicates log days duration)',

    hovermode = 'closest',

    yaxis=dict(title='Log submission count'),

    xaxis=dict(title='Submission deadline'),

)



figure=go.Figure(data=data,layout=layout)



py.iplot(figure, filename='scatter-mode')
# import relevant methods from scikit learn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



# define simple utility function to fit a model `reg` and report on scoring

def fit_and_score(reg, X, y):

    """Fit a model and score on the validation data."""

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=0)

    print('Fitting model on training datset with {:,} records and {:,} features.'.format(*X.shape))

    reg.fit(X_train, y_train)

    print('Train / validation score: {:.3f} / {:.3f}'.format(reg.score(X_train, y_train), reg.score(X_val, y_val)))

    return reg
X = usd_competitions[['ln_reward']]

y = usd_competitions.ln_submission_count



reg_reward = LinearRegression()

reg_reward = fit_and_score(reg_reward, X, y)

print('Regression coefficient [reward]: {:.2f}'.format(reg_reward.coef_[0]))
X = usd_competitions.copy()[['ln_reward', 'duration', 'MaxDailySubmissions']]

X['deadline'] = usd_competitions.deadline.dt.year + usd_competitions.deadline.dt.month / 12



reg_multifactor = LinearRegression()

reg_multifactor = fit_and_score(reg_multifactor, X, y)

print('\nRegression coefficients:')

for feature, coef in zip(X.columns, reg_multifactor.coef_):

    print('\t{:<20}: {:.2f}'.format(feature, coef))