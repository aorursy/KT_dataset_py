# Libraries
import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
projects = pd.read_csv('../input/ks-projects-201801.csv')
# Slicing just successful and failed projects
projects = projects[(projects['state'] == 'failed') | (projects['state'] == 'successful')]
projects.head()
main_colors = dict({'failed': 'rgb(200,50,50)', 'successful': 'rgb(50,200,50)'})
data = []
annotations = []

rate_success_cat = projects[projects['state'] == 'successful'].groupby(['main_category']).count()['ID']\
                / projects.groupby(['main_category']).count()['ID'] * 100
rate_failed_cat = projects[projects['state'] == 'failed'].groupby(['main_category']).count()['ID']\
                / projects.groupby(['main_category']).count()['ID'] * 100
    
rate_success_cat = rate_success_cat.sort_values(ascending=False)
rate_failed_cat = rate_failed_cat.sort_values(ascending=True)

bar_success = go.Bar(
        x=rate_success_cat.index,
        y=rate_success_cat,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=rate_failed_cat.index,
        y=rate_failed_cat,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='stack',
    title='% of successful and failed projects by main category',
    autosize=False,
    width=800,
    height=400,
    annotations=annotations
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='main_cat')
data = []

goal_success = projects[projects['state'] == 'successful'].groupby(['main_category'])\
                    .median()['usd_goal_real'].reindex(rate_success_cat.index)
goal_failed = projects[projects['state'] == 'failed'].groupby(['main_category'])\
                    .median()['usd_goal_real'].reindex(rate_success_cat.index)

bar_success = go.Bar(
        x=goal_success.index,
        y=goal_success,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=goal_failed.index,
        y=goal_failed,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='group',
    title='Median goal of successful and failed projects by main category (in USD)',
    autosize=False,
    width=800,
    height=400
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='median_goal_main_cat')
goal_dif = (goal_failed - goal_success)/goal_failed

bar_goal = go.Bar(
        x=goal_dif.index,
        y=goal_dif,
        name='failed',
        marker=dict(
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_goal]
layout = go.Layout(
    barmode='group',
    title='Relative difference of median goal of failed and successful projects (in USD)',
    autosize=False,
    width=800,
    height=400
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dif_goal_cat')
pleged_failed = projects[projects['state'] == 'failed']['usd_pledged_real']\
                        /projects[projects['state'] == 'failed']['usd_goal_real']*100
data = [go.Histogram(x=pleged_failed, marker=dict(color=main_colors['failed']))]

layout = go.Layout(
    title='% pledged of the goal amount for failed projects',
    autosize=False,
    width=800,
    height=400
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='pleged_failed')
# Calculating the length of campaign
projects['length_days'] = (pd.to_datetime(projects['deadline']) - pd.to_datetime(projects['launched'])).dt.days + 1
data = [go.Histogram(x=projects[projects['state'] == 'failed']['length_days'], 
                     marker=dict(color=main_colors['failed']),
                     name='failed'),
        go.Histogram(x=projects[projects['state'] == 'successful']['length_days'], 
                     marker=dict(color=main_colors['successful']),
                     name='successful')]

layout = go.Layout(
    barmode='stack',
    title='Campaign length distribtuion',
    autosize=False,
    width=800,
    height=400
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='length_distribution')
print('Mean days for failed projects: {0}'
      .format(round(projects[projects['state'] == 'failed']['length_days'].mean(), 2)))
print('Mean days for successful projects: {0}'
      .format(round(projects[projects['state'] == 'successful']['length_days'].mean(), 2)))
# Replacing unknown value to nan
projects['country'] = projects['country'].replace('N,0"', np.nan)

data = []
total_expected_values = []
annotations = []
shapes = []

rate_success_country = projects[projects['state'] == 'successful'].groupby(['country']).count()['ID']\
                / projects.groupby(['country']).count()['ID'] * 100
rate_failed_country = projects[projects['state'] == 'failed'].groupby(['country']).count()['ID']\
                / projects.groupby(['country']).count()['ID'] * 100
    
rate_success_country = rate_success_country.sort_values(ascending=False)
rate_failed_country = rate_failed_country.sort_values(ascending=True)

bar_success = go.Bar(
        x=rate_success_country.index,
        y=rate_success_country,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=rate_failed_country.index,
        y=rate_failed_country,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

for country in rate_success_country.index:
    weights = projects[projects['country'] == country]['main_category'].value_counts().sort_index()
    expected_values = weights * (rate_success_cat.sort_index()/100)
    total_expected_value = round(expected_values.sum() / weights.sum() * 100, 2)
    total_expected_values.append(total_expected_value)
    
for i, cat in enumerate(rate_success_country.index):
    shape = dict({
            'type': 'line',
            'x0': i-0.5,
            'y0': total_expected_values[i],
            'x1': i+1-0.5,
            'y1': total_expected_values[i],
            'line': {
                'color': 'rgb(255, 255, 255)',
                'width': 2,
            }})
    annot = dict(
            x=i,
            y=total_expected_values[i]+5,
            xref='x',
            yref='y',
            text='{0}'.format(int(total_expected_values[i])),
            font=dict({'color': 'rgb(255,255,255)'}),
            showarrow=False
        )
    annotations.append(annot)
    shapes.append(shape)

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='stack',
    title='% of successful and failed projects by country',
    autosize=False,
    width=800,
    height=400,
    annotations=annotations,
    shapes=shapes
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='main_cat')
country_medians = []
for country in rate_success_country.index:
    medians = projects[(projects['country'] == country)]\
                .groupby(['main_category']).median()['usd_goal_real'].sort_index()
    values_count = projects[(projects['country'] == country)]['main_category']\
                .value_counts().sort_index()
    median = medians[values_count > 10].mean()
    country_medians.append(median)

bar_median = go.Bar(
        x=rate_success_country.index,
        y=country_medians,
        name='failed',
        marker=dict(
            line=dict(
                width=1,
            )
        ),
    )

data = [bar_median] 
layout = go.Layout(
    barmode='group',
    title='Average median goal amount (in USD)',
    autosize=False,
    width=800,
    height=400,
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='median')
# Selecting features
projects_data = projects[['state', 'main_category', 'country', 'usd_goal_real']]

# Modifing value of dependent variable from categorical to numerical
projects_data.loc[projects_data['state'] == 'failed', 'state'] = 0
projects_data.loc[projects_data['state'] == 'successful', 'state'] = 1

# Scaling goal amount since it behaves differently in each category
for cat in projects_data['main_category'].unique():
    scaler = StandardScaler()
    new_values = scaler.fit_transform(projects_data[projects_data['main_category'] == cat][['usd_goal_real']])
    projects_data.loc[projects['main_category'] == cat, 'usd_goal_real'] = new_values.transpose()[0]

# Modifing independent variables to dummies
projects_data = pd.get_dummies(projects_data)
# Spliting data
train_X, test_X, train_y, test_y = train_test_split(projects_data.drop('state', axis=1), projects_data['state'], 
                                                    test_size=0.1, random_state=7)

# Creating model
LR = LogisticRegression()

# Fitting model
LR.fit(train_X, train_y)

# Scoring
print("Model's accuracy is {0}%".format(round(LR.score(test_X, test_y)*100, 2)))
from_largest = np.argsort(LR.coef_)[0][::-1]
positive_coef_inds = []
for index in from_largest:
    if LR.coef_[0][index] > 0:
        positive_coef_inds.append(index)
    else:
        break
print(train_X.iloc[:, positive_coef_inds].columns)
print(train_X.iloc[:, np.argmin(LR.coef_[0])].name)