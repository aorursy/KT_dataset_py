# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

data_competitions = pd.read_csv('../input/Competitions.csv');
data_compTags = pd.read_csv('../input/CompetitionTags.csv');
data_tags = pd.read_csv('../input/Tags.csv');
columns = ['Id', 'Title', 'NumScoredSubmissions', 'RewardType', 'RewardQuantity', 'TotalCompetitors', 'TotalSubmissions'];
df_comp = pd.DataFrame(data_competitions, columns=columns);
df_compTags = pd.DataFrame(data_compTags);
df_tags = pd.DataFrame(data_tags);

df_data = pd.merge(df_comp, df_compTags, left_on="Id", right_on="CompetitionId", how="left");
df_data = pd.merge(df_data, df_tags.rename(columns={'Name':'TagName'})[['Id', 'TagName']], left_on="TagId", right_on="Id", how="left").drop(['Id_x','Id_y', 'Id', 'TagId', 'CompetitionId'], axis=1);
#df_data.head()
data = df_data.groupby('TagName')[['TotalSubmissions']].sum().drop('tabular data').reset_index().sort_values('TotalSubmissions', ascending=False);
plotData = [go.Bar(x=data['TagName'], y=data['TotalSubmissions'], name="trace name here")];
layout = go.Layout(    
    title='# Competitions per Tag',
    xaxis=dict(
        title='Tag Name',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='# Competitions',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ))
fig = go.Figure(data=plotData, layout=layout);
iplot(fig);
# lets look into the reward amounts by tag
data = df_data.groupby('TagName')[['RewardQuantity']].sum().drop('tabular data').reset_index().sort_values(by="RewardQuantity", ascending=False); # we should just filter by USD reward type 

plotData = [go.Bar(x=data['TagName'], y=data['RewardQuantity'], name="trace name here")];
layout = go.Layout(    
    title='Total $ Reward per Tag',
    xaxis=dict(
        title='Tag Name',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='$ Reward',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ))
fig = go.Figure(data=plotData, layout=layout);
iplot(fig);

# if I had more time, I'd clean these up to be more accurate and dive more into how reward $ correlates to things like # submissions and possibly
# create a training set to predict # of submissions to a competiton based on Tag and reward $